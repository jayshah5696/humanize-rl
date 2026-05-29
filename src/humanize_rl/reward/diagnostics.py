"""Per-step diagnostic counters for GRPO reward functions.

Slice 2 of ``docs/plans/gemma4_rl_modal_stable_training_continuation.md``.

The reward functions in :mod:`humanize_rl.reward.grpo_rewards` call into
this module to record:

* ``record_call(...)`` \u2014 once per scoring call: total responses scored,
  per-response lengths, per-response penalty totals.
* ``record_dither()``  \u2014 each time ``dither_if_unanimous`` actually
  jitters a degenerate group.

The training callback (``humanize_rl.training.wandb_ema_callback``) calls
:func:`snapshot_and_reset` at every ``on_log`` to compute per-step
diagnostic metrics (dither_rate, penalty_rate, risk_compliance,
response_length_mean/p95).

The state is process-local; pytest can read & reset between tests via
:func:`reset`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

_LOCK = threading.Lock()


@dataclass
class _State:
    n_calls: int = 0          # total reward-scoring calls (= completions scored)
    n_dithered: int = 0       # groups that triggered dither
    n_with_penalty: int = 0   # responses with any penalty < 0
    response_lengths: list[int] = field(default_factory=list)
    penalty_totals: list[float] = field(default_factory=list)
    penalty_cap: float = 1.0  # set by build_reward_funcs; informational only


@dataclass(frozen=True)
class Snapshot:
    n_calls: int
    n_dithered: int
    n_with_penalty: int
    response_lengths: list[int]
    penalty_totals: list[float]
    penalty_cap: float


_state = _State()


def set_penalty_cap(cap: float) -> None:
    with _LOCK:
        _state.penalty_cap = float(cap)


def record_call(response_length: int, penalty_total: float) -> None:
    """Record one response. ``penalty_total`` is the (\u22640) sum of penalties."""
    with _LOCK:
        _state.n_calls += 1
        _state.response_lengths.append(int(response_length))
        _state.penalty_totals.append(float(penalty_total))
        if penalty_total < 0.0:
            _state.n_with_penalty += 1


def record_dither() -> None:
    with _LOCK:
        _state.n_dithered += 1


def snapshot_and_reset() -> Snapshot:
    """Read the current state and reset counters atomically."""
    global _state
    with _LOCK:
        snap = Snapshot(
            n_calls=_state.n_calls,
            n_dithered=_state.n_dithered,
            n_with_penalty=_state.n_with_penalty,
            response_lengths=list(_state.response_lengths),
            penalty_totals=list(_state.penalty_totals),
            penalty_cap=_state.penalty_cap,
        )
        _state = _State(penalty_cap=_state.penalty_cap)
    return snap


def reset() -> None:
    """Test helper. Drops everything except the recorded penalty_cap."""
    global _state
    with _LOCK:
        _state = _State(penalty_cap=_state.penalty_cap)
