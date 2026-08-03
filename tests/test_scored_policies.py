"""The scoring (utility) and commitment (sticky_escalation) baselines, CI-safe.

Pure V1-interface tests: a two-config catalog, hand-built runtime states, no
simulator. The properties pinned are the ones the policies were built for — soft
trade-offs for utility, and no-flapping commitment with emergency overrides for
sticky escalation.
"""

from __future__ import annotations

import pytest

from aerointentbench.policies.registry import policy_registry
from aerointentbench.policies.sticky_escalation import StickyEscalationPolicy, StickySettings
from aerointentbench.policies.utility import UtilityPolicy, UtilitySettings
from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from aerointentbench.schemas.contract import Contract, PrivacyLevel
from aerointentbench.schemas.network_trace import NetworkObservation
from aerointentbench.schemas.profile import PublicProfile, PublicProfileView, QualityTier
from aerointentbench.schemas.runtime_state import EvidenceSummary, RuntimeState

# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture
def catalog() -> ConfigCatalog:
    return ConfigCatalog(
        (
            Configuration(
                config_id="FAST",
                model_id="M_FAST",
                strategy=Strategy(placement=Placement.LOCAL, precision=Precision.FP32),
            ),
            Configuration(
                config_id="REMOTE",
                model_id="M_REMOTE",
                strategy=Strategy(
                    placement=Placement.REMOTE,
                    precision=Precision.FP32,
                    parameters={"transmitted_payload": "raw_input"},
                ),
            ),
        ),
        catalog_id="SCORED_TEST_CATALOG",
    )


@pytest.fixture
def contract() -> Contract:
    return Contract(
        contract_id="SCORED_TEST_CONTRACT",
        task_id="human_search",
        evidence_type="detection",
        quality_metric="target_recall",
        quality_operator=ComparisonOperator.GREATER_EQUAL,
        quality_threshold=0.5,
        deadline_s=60.0,
        communication_budget_mb=20.0,
        min_final_battery_frac=0.2,
        privacy_level=PrivacyLevel("remote_allowed"),
    )


def profiles() -> PublicProfileView:
    return PublicProfileView(
        {
            "FAST": PublicProfile(
                config_id="FAST",
                expected_latency_ms=400.0,
                expected_upload_mb=0.0,
                quality_tier=QualityTier("low"),
            ),
            "REMOTE": PublicProfile(
                config_id="REMOTE",
                expected_latency_ms=300.0,
                expected_upload_mb=2.0,
                quality_tier=QualityTier("high"),
            ),
        }
    )


def state(
    *,
    time_s: float = 5.0,
    battery: float = 0.9,
    comm: float = 0.0,
    progress: float = 0.1,
    bandwidth: float = 50.0,
) -> RuntimeState:
    return RuntimeState(
        current_time_s=time_s,
        frame_id=int(time_s),
        battery_frac=battery,
        power_mode="test",
        network=NetworkObservation(bandwidth_mbps=bandwidth, rtt_ms=20.0, packet_loss_frac=0.0),
        current_config_id=None,
        remaining_deadline_s=60.0 - time_s,
        cumulative_energy_j=0.0,
        cumulative_communication_mb=comm,
        path_progress=progress,
        evidence_summary=EvidenceSummary(
            predicted_unique_targets=0, processed_frames=0, mean_prediction_confidence=0.0
        ),
    )


# --- utility ---------------------------------------------------------------------------------


def test_both_policies_are_registered() -> None:
    assert "utility" in policy_registry.names()
    assert "sticky_escalation" in policy_registry.names()


def test_utility_prefers_the_higher_tier_when_costs_are_low(catalog, contract) -> None:
    policy = UtilityPolicy(public_profiles=profiles())
    assert policy.select_config(contract, state(), catalog) == "REMOTE"


def test_utility_turns_away_when_the_budget_is_nearly_gone(catalog, contract) -> None:
    policy = UtilityPolicy(public_profiles=profiles())
    broke = state(comm=0.96 * contract.communication_budget_mb)
    assert policy.select_config(contract, broke, catalog) == "FAST"


def test_utility_downshifts_under_battery_pressure(catalog, contract) -> None:
    policy = UtilityPolicy(public_profiles=profiles(), settings=UtilitySettings(w_battery=10.0))
    low = state(battery=contract.min_final_battery_frac + 0.02)
    assert policy.select_config(contract, low, catalog) == "FAST"


def test_utility_never_scores_remote_on_a_dead_link(catalog, contract) -> None:
    policy = UtilityPolicy(public_profiles=profiles())
    dead = state(bandwidth=0.0)
    assert policy.select_config(contract, dead, catalog) == "FAST"


def test_utility_is_deterministic(catalog, contract) -> None:
    policy = UtilityPolicy(public_profiles=profiles())
    picks = {policy.select_config(contract, state(), catalog) for _ in range(5)}
    assert len(picks) == 1


# --- sticky escalation -----------------------------------------------------------------------


def test_sticky_adopts_the_first_choice_immediately(catalog, contract) -> None:
    policy = StickyEscalationPolicy(public_profiles=profiles())
    assert policy.select_config(contract, state(), catalog) == "REMOTE"


def test_sticky_requires_a_dwell_before_switching(catalog, contract) -> None:
    policy = StickyEscalationPolicy(
        public_profiles=profiles(), settings=StickySettings(dwell_steps=3)
    )
    assert policy.select_config(contract, state(), catalog) == "REMOTE"  # adopt
    # The budget collapses: the scorer now wants FAST, but the incumbent holds for
    # dwell_steps decisions before the switch is allowed.
    broke = state(comm=0.96 * contract.communication_budget_mb)
    assert policy.select_config(contract, broke, catalog) == "REMOTE"  # streak 1
    assert policy.select_config(contract, broke, catalog) == "REMOTE"  # streak 2
    assert policy.select_config(contract, broke, catalog) == "FAST"  # streak 3 -> switch


def test_sticky_does_not_flap_on_alternating_preferences(catalog, contract) -> None:
    """An every-other-step challenger never accumulates the dwell streak."""
    policy = StickyEscalationPolicy(
        public_profiles=profiles(), settings=StickySettings(dwell_steps=2)
    )
    good, broke = state(), state(comm=0.96 * contract.communication_budget_mb)
    assert policy.select_config(contract, good, catalog) == "REMOTE"
    for _ in range(4):  # desired alternates FAST/REMOTE; incumbent must hold
        assert policy.select_config(contract, broke, catalog) == "REMOTE"
        assert policy.select_config(contract, good, catalog) == "REMOTE"


def test_sticky_battery_emergency_bypasses_the_dwell(catalog, contract) -> None:
    policy = StickyEscalationPolicy(
        public_profiles=profiles(), settings=StickySettings(dwell_steps=99)
    )
    assert policy.select_config(contract, state(), catalog) == "REMOTE"
    critical = state(battery=contract.min_final_battery_frac + 0.01)
    assert policy.select_config(contract, critical, catalog) == "FAST"  # immediate


def test_sticky_abandons_an_unreachable_remote_incumbent(catalog, contract) -> None:
    policy = StickyEscalationPolicy(
        public_profiles=profiles(), settings=StickySettings(dwell_steps=99)
    )
    assert policy.select_config(contract, state(), catalog) == "REMOTE"
    dead = state(bandwidth=0.0)
    assert policy.select_config(contract, dead, catalog) == "FAST"  # immediate


def test_sticky_reset_clears_the_incumbent(catalog, contract) -> None:
    policy = StickyEscalationPolicy(public_profiles=profiles())
    policy.select_config(contract, state(), catalog)
    policy.reset()
    assert policy.select_config(contract, state(), catalog) == "REMOTE"
