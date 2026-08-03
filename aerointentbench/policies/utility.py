"""A scoring baseline: soft cost-benefit trade-off instead of lexicographic filters.

``rule_based`` decides by *elimination* (privacy → reachability → budget → pressure →
latency → best tier): each filter is absolute, so a configuration that is slightly
worse on one axis but much better on another can never win. This policy replaces the
filters with a single scalar **utility** per configuration and picks the argmax:

    score(c) = w_quality * tier(c)
             - w_skip    * frames_lost(c)          # latency beyond one decision interval
             - w_comm    * upload(c) / spendable   # share of the remaining budget consumed
             - w_battery * pressure * latency(c)/interval   # latency as the public energy proxy,
                                                            # scaled by how close the floor is

Hard exclusions stay hard — privacy-forbidden configurations and remote configurations
on a dead link are not scored at all, because selecting them is a recorded violation or
a guaranteed failure, not a trade-off.

Epistemic position unchanged from V1: inputs are the contract, the frozen
``RuntimeState``, and public profiles; ``latency`` is the estimate a deployed system
could make (upload/bandwidth + RTT + disclosed compute), and energy is *not* public,
so battery pressure can only scale the latency proxy — exactly the limitation
``rule_based`` documents. With profiles hidden the score degenerates to catalog order,
degrading rather than failing.

The weights are named settings, not magic numbers; they are a baseline's assumptions,
not tuned-to-a-family values, and the multi-seed harness is the place that keeps that
claim honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from aerointentbench.policies.base import estimated_latency_s
from aerointentbench.schemas.configuration import ConfigCatalog, Configuration, Placement
from aerointentbench.schemas.contract import Contract
from aerointentbench.schemas.profile import PublicProfileView, QualityTier
from aerointentbench.schemas.runtime_state import RuntimeState
from aerointentbench.simulator.action_validator import privacy_permits

__all__ = ["UtilityPolicy", "UtilitySettings"]

_DEFAULT_TIER_RANK: Final = -1


@dataclass(frozen=True, slots=True)
class UtilitySettings:
    """Named trade-off weights. Units: utility per tier rank; costs are dimensionless."""

    w_quality: float = 1.0
    #: Cost per decision frame the configuration's latency would skip.
    w_skip: float = 0.6
    #: Cost of consuming the entire remaining spendable communication budget at once.
    w_comm: float = 0.8
    #: Cost of one interval of latency when the battery floor is imminent.
    w_battery: float = 2.0
    #: Battery headroom (above the contract floor) below which pressure rises from 0
    #: toward 1; at the floor itself the pressure is exactly 1.
    battery_pressure_band_frac: float = 0.15
    #: Fraction of the communication budget never counted as spendable.
    communication_reserve_frac: float = 0.05
    decision_interval_s: float = 1.0


class UtilityPolicy:
    """Argmax of a scalar utility over the permitted, reachable configurations."""

    __slots__ = ("_profiles", "_settings")

    def __init__(
        self,
        *,
        public_profiles: PublicProfileView | None = None,
        settings: UtilitySettings | None = None,
    ) -> None:
        self._profiles = (
            public_profiles if public_profiles is not None else PublicProfileView.hidden()
        )
        self._settings = settings or UtilitySettings()

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        candidates = [
            config for config in configs if privacy_permits(contract.privacy_level, config)
        ]
        if not candidates:
            return next(iter(configs)).config_id  # impossible pool; the validator records it
        scored = [
            (self.score(config, contract, state), index, config)
            for index, config in enumerate(candidates)
            if not self._excluded(config, state)
        ]
        if not scored:  # every candidate is an unreachable remote: fall back to the pool
            scored = [
                (self.score(config, contract, state), index, config)
                for index, config in enumerate(candidates)
            ]
        # Highest score wins; ties break toward lower estimated latency, then catalog
        # order — all deterministic.
        best = max(
            scored, key=lambda entry: (entry[0], -self._latency_key(entry[2], state), -entry[1])
        )
        return best[2].config_id

    # -- scoring ------------------------------------------------------------------------

    def score(self, config: Configuration, contract: Contract, state: RuntimeState) -> float:
        """The utility of running ``config`` now. Public information only."""
        settings = self._settings
        profile = self._profiles.get(config.config_id)
        tier = _DEFAULT_TIER_RANK if profile is None else QualityTier(profile.quality_tier).rank
        latency = estimated_latency_s(config, profile, state.network)
        latency_intervals = 1.0 if latency is None else latency / settings.decision_interval_s
        frames_lost = max(0.0, latency_intervals - 1.0)

        upload = 0.0 if profile is None else profile.expected_upload_mb
        spendable = (
            contract.communication_budget_mb * (1.0 - settings.communication_reserve_frac)
            - state.cumulative_communication_mb
        )
        if upload <= 0.0:
            comm_cost = 0.0
        elif spendable <= 0.0:
            comm_cost = float("inf")  # any upload now guarantees a budget breach
        else:
            comm_cost = upload / spendable

        headroom = state.battery_frac - contract.min_final_battery_frac
        band = settings.battery_pressure_band_frac
        pressure = min(1.0, max(0.0, (band - headroom) / band)) if band > 0 else 0.0

        return (
            settings.w_quality * tier
            - settings.w_skip * frames_lost
            - settings.w_comm * comm_cost
            - settings.w_battery * pressure * latency_intervals
        )

    # -- exclusions ---------------------------------------------------------------------

    def _excluded(self, config: Configuration, state: RuntimeState) -> bool:
        """Remote on a dead link is a guaranteed failure, not a trade-off."""
        return config.strategy.placement is Placement.REMOTE and state.network.is_disconnected

    def _latency_key(self, config: Configuration, state: RuntimeState) -> float:
        latency = estimated_latency_s(config, self._profiles.get(config.config_id), state.network)
        return float("inf") if latency is None else latency

    def __repr__(self) -> str:
        return f"UtilityPolicy(profiles_available={self._profiles.is_available})"
