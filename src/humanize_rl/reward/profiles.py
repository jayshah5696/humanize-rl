"""Reward profile weights for humanize RL tasks."""

from __future__ import annotations

from dataclasses import dataclass

from humanize_rl.reward.tasks import RewardProfileName, RLTask


@dataclass(frozen=True)
class RewardProfile:
    """Named component weights for one reward profile."""

    name: RewardProfileName
    weights: dict[str, float]

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())


PROFILES: dict[RewardProfileName, RewardProfile] = {
    "rewrite_faithful_concise": RewardProfile(
        name="rewrite_faithful_concise",
        weights={
            "style": 0.30,
            "faithfulness": 0.30,
            "task_following": 0.20,
            "length": 0.10,
            "format": 0.10,
        },
    ),
    "direct_workplace_message": RewardProfile(
        name="direct_workplace_message",
        weights={
            "task_following": 0.35,
            "style": 0.25,
            "format": 0.20,
            "length": 0.10,
            "placeholder": 0.10,
        },
    ),
    "compression_update": RewardProfile(
        name="compression_update",
        weights={
            "fact_preservation": 0.30,
            "length": 0.25,
            "clarity": 0.20,
            "style": 0.15,
            "format": 0.10,
        },
    ),
    "technical_explain_natural": RewardProfile(
        name="technical_explain_natural",
        weights={
            "correctness_adherence": 0.30,
            "clarity": 0.25,
            "naturalness": 0.20,
            "structure_restraint": 0.15,
            "length": 0.10,
        },
    ),
    "sensitive_comms": RewardProfile(
        name="sensitive_comms",
        weights={
            "task_following": 0.30,
            "tone_appropriateness": 0.25,
            "concision": 0.20,
            "no_corporate_filler": 0.15,
            "format": 0.10,
        },
    ),
}


def get_profile(name: RewardProfileName) -> RewardProfile:
    """Return a reward profile by name."""
    return PROFILES[name]


def profile_for_task(task: RLTask) -> RewardProfile:
    """Return the configured reward profile for a task."""
    return get_profile(task.reward_profile)
