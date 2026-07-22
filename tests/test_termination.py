"""Episode termination rules."""

from __future__ import annotations

import dataclasses

import pytest

from aerointentbench.simulator.battery_model import BatteryState, SimpleBatteryModel
from aerointentbench.simulator.path import ConstantVelocityPath
from aerointentbench.simulator.state_manager import SimulationState, StateManager
from aerointentbench.simulator.termination import (
    DEFAULT_TERMINATION_CONDITIONS,
    BatteryDepleted,
    DeadlineExceeded,
    PathComplete,
    TerminationReason,
    first_triggered,
)


@pytest.fixture
def running_state(state_manager: StateManager) -> SimulationState:
    return state_manager.initial_state()


def _at(state: SimulationState, **overrides) -> SimulationState:
    return dataclasses.replace(state, **overrides)


# --- individual conditions ------------------------------------------------------------


def test_a_fresh_episode_does_not_terminate(running_state, synthetic_contract) -> None:
    assert first_triggered(DEFAULT_TERMINATION_CONDITIONS, running_state, synthetic_contract) is None


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0.99, None), (1.0, TerminationReason.PATH_COMPLETE), (1.5, TerminationReason.PATH_COMPLETE)],
)
def test_path_completion(running_state, synthetic_contract, progress: float, expected) -> None:
    assert PathComplete().check(_at(running_state, path_progress=progress), synthetic_contract) == expected


@pytest.mark.parametrize(
    ("time_s", "expected"),
    [
        (59.9, None),
        (60.0, None),  # finishing exactly on the deadline is on time
        (60.001, TerminationReason.DEADLINE_EXCEEDED),
        (90.0, TerminationReason.DEADLINE_EXCEEDED),
    ],
)
def test_deadline_is_exceeded_only_strictly_past_it(
    running_state, synthetic_contract, time_s: float, expected
) -> None:
    assert DeadlineExceeded().check(_at(running_state, current_time_s=time_s), synthetic_contract) == expected


@pytest.mark.parametrize(
    ("remaining_j", "expected"),
    [(1.0, None), (0.0, TerminationReason.BATTERY_DEPLETED)],
)
def test_battery_depletion(running_state, synthetic_contract, remaining_j: float, expected) -> None:
    state = _at(running_state, battery=BatteryState(capacity_j=360_000.0, remaining_j=remaining_j))
    assert BatteryDepleted().check(state, synthetic_contract) == expected


# --- composition ----------------------------------------------------------------------


def test_default_order_reports_path_completion_first(running_state, synthetic_contract) -> None:
    """A step that both finished the path and drained the battery did finish the path."""
    state = _at(
        running_state,
        path_progress=1.0,
        current_time_s=999.0,
        battery=BatteryState(capacity_j=360_000.0, remaining_j=0.0),
    )
    assert (
        first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract)
        is TerminationReason.PATH_COMPLETE
    )


def test_deadline_is_reported_before_battery(running_state, synthetic_contract) -> None:
    state = _at(
        running_state,
        current_time_s=61.0,
        battery=BatteryState(capacity_j=360_000.0, remaining_j=0.0),
    )
    assert (
        first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract)
        is TerminationReason.DEADLINE_EXCEEDED
    )


def test_termination_is_deterministic(running_state, synthetic_contract) -> None:
    state = _at(running_state, path_progress=1.0)
    reasons = {first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract) for _ in range(5)}
    assert reasons == {TerminationReason.PATH_COMPLETE}


def test_conditions_are_composable_without_changing_the_core(running_state, synthetic_contract) -> None:
    """A new condition is an addition, not another branch in the runner."""

    class AlwaysStop:
        def check(self, state, synthetic_contract):
            del state, synthetic_contract
            return TerminationReason.PATH_COMPLETE

    assert first_triggered((), running_state, synthetic_contract) is None
    assert first_triggered((AlwaysStop(),), running_state, synthetic_contract) is not None


def test_an_episode_reaches_path_completion_by_flying_it(
    state_manager: StateManager, synthetic_contract
) -> None:
    """End to end over the transition: 50 steps of 1 s completes the 250 m path."""
    state = state_manager.initial_state()
    steps = 0
    while first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract) is None:
        state = state_manager.advance(
            state,
            selected_config_id="CFG_LOCAL_LIGHT",
            inference_latency_s=0.1,
            onboard_energy_j=3.0,
            communication_mb=0.0,
        )
        steps += 1
        assert steps < 200, "the loop must terminate"

    assert steps == 50
    assert state.current_time_s == 50.0
    assert (
        first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract)
        is TerminationReason.PATH_COMPLETE
    )


def _fly_until_termination(
    manager: StateManager, synthetic_contract, *, latency_s: float
) -> SimulationState:
    state = manager.initial_state()
    while first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract) is None:
        state = manager.advance(
            state,
            selected_config_id="CFG_LOCAL_STRONG",
            inference_latency_s=latency_s,
            onboard_energy_j=12.0,
            communication_mb=0.0,
        )
        assert state.step_index < 500, "the loop must terminate"
    return state


def test_step_length_does_not_change_when_the_path_is_flown(
    state_manager: StateManager, synthetic_contract
) -> None:
    """The vehicle covers ground on wall-clock time, so a slow policy does not fly slower.

    What a slow configuration costs is frames, not distance. Path completion lands on the
    path's own duration, give or take the overshoot of whichever step straddles it.
    """
    state = _fly_until_termination(state_manager, synthetic_contract, latency_s=5.0)
    assert state.current_time_s == 50.0  # 250 m at 5 m/s, reached exactly by 5 s steps
    assert state.step_index == 10


def test_a_long_step_can_overshoot_the_path_end_past_the_deadline(
    state_manager: StateManager, synthetic_contract
) -> None:
    """Termination reason and constraint success are computed separately, and can disagree.

    16 s steps land on 64 s: the step that straddles the 50 s path end carries the clock
    past the 60 s deadline. The episode still stops for PATH_COMPLETE -- it did complete
    the path -- while the deadline metric, computed from the final time, fails.
    """
    state = _fly_until_termination(state_manager, synthetic_contract, latency_s=16.0)

    assert state.current_time_s == 64.0
    assert state.path_progress == 1.0
    assert (
        first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract)
        is TerminationReason.PATH_COMPLETE
    )
    assert state.current_time_s > synthetic_contract.deadline_s, "the deadline constraint fails"


def test_deadline_termination_fires_when_the_path_outlasts_the_deadline(
    episode, platform, synthetic_contract
) -> None:
    """A path that cannot be flown inside the deadline stops for DEADLINE_EXCEEDED."""
    manager = StateManager(
        episode=episode,
        platform=platform,
        # 500 m at 5 m/s takes 100 s, well past the 60 s deadline.
        path=ConstantVelocityPath(length_m=500.0, velocity_mps=5.0),
        battery_model=SimpleBatteryModel(),
    )
    state = _fly_until_termination(manager, synthetic_contract, latency_s=1.0)

    assert state.current_time_s == 61.0
    assert state.path_progress < 1.0
    assert (
        first_triggered(DEFAULT_TERMINATION_CONDITIONS, state, synthetic_contract)
        is TerminationReason.DEADLINE_EXCEEDED
    )
