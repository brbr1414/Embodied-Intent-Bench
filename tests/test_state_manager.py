"""Internal state transitions and the projection onto the policy-visible observation."""

from __future__ import annotations

import dataclasses

import pytest

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.schemas.network_trace import NetworkObservation
from aerointentbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from aerointentbench.simulator.battery_model import SimpleBatteryModel
from aerointentbench.simulator.path import ConstantVelocityPath
from aerointentbench.simulator.state_manager import SimulationState, StateManager

SUMMARY = EvidenceSummary(
    predicted_unique_targets=3, processed_frames=20, mean_prediction_confidence=0.78
)
NETWORK = NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.01)


def _advance(manager: StateManager, state: SimulationState, **overrides) -> SimulationState:
    kwargs = {
        "selected_config_id": "CFG_LOCAL_LIGHT",
        "inference_latency_s": 0.1,
        "onboard_energy_j": 3.0,
        "communication_mb": 0.0,
    }
    return manager.advance(state, **{**kwargs, **overrides})


# --- initial state ------------------------------------------------------------------


def test_initial_state_starts_at_zero(state_manager: StateManager, episode) -> None:
    state = state_manager.initial_state()
    assert state.current_time_s == 0.0
    assert state.frame_id == 0
    assert state.step_index == 0
    assert state.path_progress == 0.0
    assert state.cumulative_communication_mb == 0.0
    assert state.cumulative_energy.total_j == 0.0
    assert state.battery.fraction == pytest.approx(episode.initial_battery_frac)
    assert state.power_mode == episode.power_mode


def test_initial_configuration_comes_from_the_episode(state_manager: StateManager) -> None:
    assert state_manager.initial_state().current_config_id is None


def test_unsupported_power_mode_fails_at_construction(episode, platform, path_spec) -> None:
    """A specification error, caught before the episode starts rather than mid-run."""
    broken = dataclasses.replace(episode, power_mode="500W")
    with pytest.raises(SchemaValidationError, match="does not support power mode"):
        StateManager(
            episode=broken,
            platform=platform,
            path=ConstantVelocityPath.from_spec(path_spec, velocity_mps=5.0),
            battery_model=SimpleBatteryModel(),
        )


# --- advancing ----------------------------------------------------------------------


def test_a_fast_step_still_advances_one_second(state_manager: StateManager) -> None:
    after = _advance(state_manager, state_manager.initial_state(), inference_latency_s=0.1)
    assert after.current_time_s == 1.0
    assert after.frame_id == 1
    assert after.step_index == 1


def test_a_slow_step_advances_by_its_latency_and_skips_frames(
    state_manager: StateManager,
) -> None:
    after = _advance(state_manager, state_manager.initial_state(), inference_latency_s=3.2)
    assert after.current_time_s == pytest.approx(3.2)
    assert after.frame_id == 3, "frames 1 and 2 elapsed unprocessed"


def test_advance_does_not_mutate_the_previous_state(state_manager: StateManager) -> None:
    before = state_manager.initial_state()
    _advance(state_manager, before)
    assert before.current_time_s == 0.0
    assert before.step_index == 0


def test_energy_and_communication_accumulate(state_manager: StateManager) -> None:
    state = state_manager.initial_state()
    state = _advance(state_manager, state, communication_mb=1.5)
    state = _advance(state_manager, state, communication_mb=1.5)

    assert state.cumulative_communication_mb == 3.0
    # Two 1 s steps: flight 2*180, compute 2*3, communication 3 MB * 0.5 J/MB.
    assert state.cumulative_energy.flight_j == 360.0
    assert state.cumulative_energy.compute_j == 6.0
    assert state.cumulative_energy.communication_j == 1.5
    assert state.cumulative_energy.total_j == pytest.approx(367.5)


def test_battery_drains_through_the_injected_model(state_manager: StateManager) -> None:
    state = state_manager.initial_state()
    start_j = state.battery.remaining_j
    state = _advance(state_manager, state)
    assert state.battery.remaining_j == start_j - 183.0


def test_path_progress_tracks_the_clock(state_manager: StateManager) -> None:
    state = state_manager.initial_state()
    for _ in range(25):
        state = _advance(state_manager, state)
    # 25 s of a 50 s path.
    assert state.current_time_s == 25.0
    assert state.path_progress == pytest.approx(0.5)


def test_the_executed_configuration_becomes_the_current_one(state_manager: StateManager) -> None:
    state = _advance(
        state_manager, state_manager.initial_state(), selected_config_id="CFG_REMOTE_STRONG"
    )
    assert state.current_config_id == "CFG_REMOTE_STRONG"


def test_a_failed_execution_still_costs_time_and_flight_energy(
    state_manager: StateManager,
) -> None:
    """The vehicle flies on when inference fails; only compute and comms drop to zero."""
    state = _advance(
        state_manager,
        state_manager.initial_state(),
        inference_latency_s=2.0,
        onboard_energy_j=0.0,
        communication_mb=0.0,
    )
    assert state.current_time_s == 2.0
    assert state.cumulative_energy.flight_j == 360.0
    assert state.cumulative_energy.compute_j == 0.0
    assert state.cumulative_communication_mb == 0.0


def test_stepping_is_deterministic(state_manager: StateManager) -> None:
    first = _advance(state_manager, state_manager.initial_state())
    second = _advance(state_manager, state_manager.initial_state())
    assert first == second


# --- policy-visible projection --------------------------------------------------------


def _runtime_state(manager: StateManager, state: SimulationState, synthetic_contract) -> RuntimeState:
    return manager.build_runtime_state(state, contract=synthetic_contract, network=NETWORK, evidence_summary=SUMMARY)


def test_runtime_state_reflects_internal_state(state_manager: StateManager, synthetic_contract) -> None:
    state = state_manager.initial_state()
    for _ in range(25):
        state = _advance(state_manager, state, communication_mb=0.78)

    observation = _runtime_state(state_manager, state, synthetic_contract)

    assert observation.current_time_s == 25.0
    assert observation.frame_id == 25
    assert observation.path_progress == pytest.approx(0.5)
    assert observation.network is NETWORK
    assert observation.evidence_summary is SUMMARY
    assert observation.cumulative_communication_mb == pytest.approx(19.5)
    assert observation.battery_frac == pytest.approx(state.battery.fraction)
    assert observation.cumulative_energy_j == pytest.approx(state.cumulative_energy.total_j)


def test_remaining_deadline_counts_down(state_manager: StateManager, synthetic_contract) -> None:
    state = state_manager.initial_state()
    for _ in range(25):
        state = _advance(state_manager, state)
    assert _runtime_state(state_manager, state, synthetic_contract).remaining_deadline_s == 35.0


def test_remaining_deadline_clamps_at_zero(state_manager: StateManager, synthetic_contract) -> None:
    """A negative "remaining" would be a nonsense reading; the overrun shows in the clock."""
    state = state_manager.initial_state()
    state = _advance(state_manager, state, inference_latency_s=90.0)
    observation = _runtime_state(state_manager, state, synthetic_contract)
    assert observation.current_time_s == 90.0
    assert observation.remaining_deadline_s == 0.0


def test_the_observation_carries_no_simulator_handle(
    state_manager: StateManager, synthetic_contract
) -> None:
    """The policy gets a projection, not the simulator's own state object."""
    state = state_manager.initial_state()
    observation = _runtime_state(state_manager, state, synthetic_contract)
    assert isinstance(observation, RuntimeState)
    values = [getattr(observation, field.name) for field in dataclasses.fields(observation)]
    assert not any(isinstance(value, (SimulationState, StateManager)) for value in values)


def test_the_observation_is_immutable(state_manager: StateManager, synthetic_contract) -> None:
    observation = _runtime_state(state_manager, state_manager.initial_state(), synthetic_contract)
    with pytest.raises(AttributeError):
        observation.battery_frac = 0.0  # type: ignore[misc]


def test_battery_model_is_replaceable_without_touching_the_state_manager(
    episode, platform, path_spec
) -> None:
    """Extensibility criterion 4: swap the battery model, change nothing else."""

    class NeverDrainsBatteryModel:
        def transition(self, previous_state, usage, elapsed_time_s):
            del usage, elapsed_time_s
            return previous_state

    manager = StateManager(
        episode=episode,
        platform=platform,
        path=ConstantVelocityPath.from_spec(path_spec, velocity_mps=episode.velocity_mps),
        battery_model=NeverDrainsBatteryModel(),
    )
    state = _advance(manager, manager.initial_state())

    assert state.battery.fraction == pytest.approx(episode.initial_battery_frac)
    # The ledger still records what was spent, even though this model ignores it.
    assert state.cumulative_energy.total_j == 183.0
