"""A commitment baseline: utility-driven choices with switching hysteresis.

The V3 P3 evaluation exposed a concrete failure mode: the projecting
``budget_planner`` *flapped* — alternating strong and light almost every step — which
halved the strong model's effective cadence and missed the late targets, losing to
the cruder ``rule_based`` that happened to hold its choice through the decisive
window. The lesson is that in a closed loop with latency-driven frame skipping,
**switching itself has a cost** that per-step argmax policies cannot see.

This policy makes that cost explicit through commitment: it computes the same scalar
utility as :class:`~aerointentbench.policies.utility.UtilityPolicy`, but switches
away from the incumbent configuration only when a challenger has out-scored it for
``dwell_steps`` consecutive decisions. Two emergencies bypass the dwell, because
waiting would convert a recoverable state into a contract violation:

- **battery emergency** — headroom over the contract floor below the emergency band
  switches to the fastest candidate immediately;
- **dead link** — an incumbent remote configuration on a disconnected network is a
  guaranteed failure, so it is abandoned immediately.

Stateful within one episode (incumbent + challenger streak), like the other stateful
baselines; the composition root constructs policies per run, and ``reset()`` exists
for reuse. Inputs remain the contract, the frozen ``RuntimeState``, and public
profiles — nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aerointentbench.policies.base import estimated_latency_s
from aerointentbench.policies.utility import UtilityPolicy, UtilitySettings
from aerointentbench.schemas.configuration import ConfigCatalog, Configuration, Placement
from aerointentbench.schemas.contract import Contract
from aerointentbench.schemas.profile import PublicProfileView
from aerointentbench.schemas.runtime_state import RuntimeState

__all__ = ["StickyEscalationPolicy", "StickySettings"]


@dataclass(frozen=True, slots=True)
class StickySettings:
    """Commitment parameters on top of the utility weights."""

    #: Consecutive decisions a challenger must win before the policy switches.
    dwell_steps: int = 3
    #: Battery headroom below which the emergency (immediate fastest-local) path fires.
    emergency_battery_band_frac: float = 0.04
    utility: UtilitySettings = field(default_factory=UtilitySettings)


class StickyEscalationPolicy:
    """Utility argmax with a dwell requirement before any non-emergency switch."""

    __slots__ = ("_challenger", "_incumbent", "_scorer", "_settings", "_streak")

    def __init__(
        self,
        *,
        public_profiles: PublicProfileView | None = None,
        settings: StickySettings | None = None,
    ) -> None:
        self._settings = settings or StickySettings()
        self._scorer = UtilityPolicy(
            public_profiles=public_profiles, settings=self._settings.utility
        )
        self._incumbent: str | None = None
        self._challenger: str | None = None
        self._streak = 0

    def reset(self) -> None:
        """Forget the incumbent and streak (between episodes, if an instance is reused)."""
        self._incumbent = None
        self._challenger = None
        self._streak = 0

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        desired = self._scorer.select_config(contract, state, configs)

        if self._incumbent is None or self._incumbent not in configs:
            return self._adopt(desired)

        if self._battery_emergency(contract, state):
            return self._adopt(self._fastest_local(configs, state))
        if self._incumbent_unreachable(configs, state):
            return self._adopt(desired)

        if desired == self._incumbent:
            self._challenger, self._streak = None, 0
            return self._incumbent

        # A challenger must hold its lead for dwell_steps consecutive decisions.
        if desired == self._challenger:
            self._streak += 1
        else:
            self._challenger, self._streak = desired, 1
        if self._streak >= self._settings.dwell_steps:
            return self._adopt(desired)
        return self._incumbent

    # -- internals ----------------------------------------------------------------------

    def _adopt(self, config_id: str) -> str:
        self._incumbent = config_id
        self._challenger, self._streak = None, 0
        return config_id

    def _battery_emergency(self, contract: Contract, state: RuntimeState) -> bool:
        headroom = state.battery_frac - contract.min_final_battery_frac
        return headroom < self._settings.emergency_battery_band_frac

    def _incumbent_unreachable(self, configs: ConfigCatalog, state: RuntimeState) -> bool:
        if not state.network.is_disconnected or self._incumbent is None:
            return False
        incumbent = configs.get(self._incumbent)
        return incumbent.strategy.placement is Placement.REMOTE

    def _fastest_local(self, configs: ConfigCatalog, state: RuntimeState) -> str:
        def latency_key(config: Configuration) -> float:
            latency = estimated_latency_s(
                config, self._scorer._profiles.get(config.config_id), state.network
            )
            return float("inf") if latency is None else latency

        local = [c for c in configs if c.strategy.placement is Placement.LOCAL]
        pool = local or list(configs)
        return min(pool, key=latency_key).config_id

    def __repr__(self) -> str:
        return (
            f"StickyEscalationPolicy(incumbent={self._incumbent!r}, "
            f"dwell={self._settings.dwell_steps})"
        )
