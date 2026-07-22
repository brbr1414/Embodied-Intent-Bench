"""Aggregate metrics across an episode suite.

**Mission Success Rate is the primary metric**: the fraction of episodes in which every hard
constraint held. The remaining rates decompose it -- when a policy's success rate is low,
they say which constraint it kept breaking.

Every mean is an unweighted mean *over episodes*, so each episode counts once regardless of
how many steps it ran. A per-step weighting would let a long episode dominate the average
latency, which would report something about episode length rather than about the policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from aerointentbench.metrics.episode_metrics import EpisodeMetrics

__all__ = ["AggregateMetrics", "aggregate_metrics"]


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """A policy's scored result over a suite."""

    episode_count: int

    mission_success_rate: float
    quality_success_rate: float
    deadline_success_rate: float
    battery_constraint_success_rate: float
    communication_constraint_success_rate: float
    privacy_constraint_success_rate: float

    mean_final_battery_fraction: float
    mean_mission_completion_time_s: float
    mean_end_to_end_inference_latency_ms: float
    mean_communication_mb: float
    mean_total_energy_j: float
    mean_configuration_switch_count: float
    mean_constraint_violation_count: float
    mean_quality_value: float

    total_invalid_action_count: int
    total_failed_inference_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "mission_success_rate": self.mission_success_rate,
            "rates": {
                "quality_success_rate": self.quality_success_rate,
                "deadline_success_rate": self.deadline_success_rate,
                "battery_constraint_success_rate": self.battery_constraint_success_rate,
                "communication_constraint_success_rate": self.communication_constraint_success_rate,
                "privacy_constraint_success_rate": self.privacy_constraint_success_rate,
            },
            "means": {
                "mean_quality_value": self.mean_quality_value,
                "mean_final_battery_fraction": self.mean_final_battery_fraction,
                "mean_mission_completion_time_s": self.mean_mission_completion_time_s,
                "mean_end_to_end_inference_latency_ms": self.mean_end_to_end_inference_latency_ms,
                "mean_communication_mb": self.mean_communication_mb,
                "mean_total_energy_j": self.mean_total_energy_j,
                "mean_configuration_switch_count": self.mean_configuration_switch_count,
                "mean_constraint_violation_count": self.mean_constraint_violation_count,
            },
            "totals": {
                "total_invalid_action_count": self.total_invalid_action_count,
                "total_failed_inference_count": self.total_failed_inference_count,
            },
        }


def aggregate_metrics(episodes: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate per-episode metrics over a suite.

    An empty suite yields zeros rather than raising: reporting "no episodes ran" as a result
    is more useful than an exception halfway through a campaign.
    """
    count = len(episodes)
    if count == 0:
        return AggregateMetrics(
            episode_count=0,
            **{field: 0.0 for field in _RATE_FIELDS},
            **{field: 0.0 for field in _MEAN_FIELDS},
            total_invalid_action_count=0,
            total_failed_inference_count=0,
        )

    def rate(attribute: str) -> float:
        return sum(bool(getattr(episode, attribute)) for episode in episodes) / count

    def mean(attribute: str) -> float:
        return sum(getattr(episode, attribute) for episode in episodes) / count

    return AggregateMetrics(
        episode_count=count,
        mission_success_rate=rate("mission_success"),
        quality_success_rate=rate("quality_success"),
        deadline_success_rate=rate("deadline_success"),
        battery_constraint_success_rate=rate("battery_constraint_success"),
        communication_constraint_success_rate=rate("communication_constraint_success"),
        privacy_constraint_success_rate=rate("privacy_constraint_success"),
        mean_final_battery_fraction=mean("final_battery_fraction"),
        mean_mission_completion_time_s=mean("mission_completion_time_s"),
        mean_end_to_end_inference_latency_ms=mean("mean_end_to_end_inference_latency_ms"),
        mean_communication_mb=mean("total_communication_mb"),
        mean_total_energy_j=mean("total_energy_j"),
        mean_configuration_switch_count=mean("configuration_switch_count"),
        mean_constraint_violation_count=mean("constraint_violation_count"),
        mean_quality_value=mean("quality_value"),
        total_invalid_action_count=sum(episode.invalid_action_count for episode in episodes),
        total_failed_inference_count=sum(episode.failed_inference_count for episode in episodes),
    )


_RATE_FIELDS = (
    "mission_success_rate",
    "quality_success_rate",
    "deadline_success_rate",
    "battery_constraint_success_rate",
    "communication_constraint_success_rate",
    "privacy_constraint_success_rate",
)
_MEAN_FIELDS = (
    "mean_final_battery_fraction",
    "mean_mission_completion_time_s",
    "mean_end_to_end_inference_latency_ms",
    "mean_communication_mb",
    "mean_total_energy_j",
    "mean_configuration_switch_count",
    "mean_constraint_violation_count",
    "mean_quality_value",
)
