"""The decision loop, and the composition root that assembles it."""

from __future__ import annotations

from pathlib import Path

import pytest

from aerointentbench.benchmark import BenchmarkData, run_episode, run_suite
from aerointentbench.executor.base import ExecutionResult, FailureReason
from aerointentbench.policies import StaticPolicy
from aerointentbench.simulator.termination import TerminationReason


@pytest.fixture(scope="module")
def data(request) -> BenchmarkData:
    return BenchmarkData(Path(request.config.rootdir) / "data")


@pytest.fixture
def first_episode(data: BenchmarkData):
    return next(e for e in data.episodes() if e.episode_id == "EPISODE_001")


# --- the loop ------------------------------------------------------------------------


def test_an_episode_runs_to_termination(data, first_episode, contract) -> None:
    result = run_episode(data=data, episode=first_episode, contract=contract)
    record = result.record

    assert record.termination_reason is TerminationReason.PATH_COMPLETE
    assert record.final_path_progress == 1.0
    assert record.final_time_s == pytest.approx(900.0)
    assert record.step_count == 900


def test_every_step_is_recorded(data, first_episode, contract) -> None:
    record = run_episode(data=data, episode=first_episode, contract=contract).record
    step = record.steps[0]

    assert step.step_index == 0
    assert step.time_s == 0.0
    assert set(step.state) >= {"battery_frac", "network", "path_progress", "evidence_summary"}
    assert step.execution["success"] is True
    assert step.executed_config_id in data.catalog


def test_the_step_log_captures_what_the_policy_actually_saw(data, first_episode, contract) -> None:
    record = run_episode(data=data, episode=first_episode, contract=contract).record
    early, late = record.steps[0], record.steps[-1]

    assert early.state["battery_frac"] > late.state["battery_frac"]
    assert early.state["path_progress"] < late.state["path_progress"]
    assert late.state["cumulative_communication_mb"] >= early.state["cumulative_communication_mb"]


def test_replaying_an_episode_reproduces_it_exactly(data, first_episode, contract) -> None:
    """Determinism is a benchmark requirement, not a convenience."""
    first = run_episode(data=data, episode=first_episode, contract=contract)
    second = run_episode(data=data, episode=first_episode, contract=contract)

    assert first.record.to_dict(include_detail=True) == second.record.to_dict(include_detail=True)
    assert first.metrics.to_dict() == second.metrics.to_dict()


def test_a_policy_that_raises_does_not_take_the_benchmark_with_it(
    data, first_episode, contract
) -> None:
    """A submitted policy misbehaving is a result to report, not a crash."""

    class ExplodingPolicy:
        def select_config(self, contract, state, configs):
            raise RuntimeError("boom")

    result = run_episode(
        data=data, episode=first_episode, contract=contract, policy=ExplodingPolicy()
    )
    assert result.record.invalid_action_count == result.record.step_count
    assert result.record.step_count > 0
    assert all(step.executed_config_id == "CFG_LOCAL_LIGHT" for step in result.record.steps)


def test_an_invalid_action_is_substituted_and_counted(data, first_episode, contract) -> None:
    result = run_episode(
        data=data, episode=first_episode, contract=contract, policy=StaticPolicy("CFG_NOT_REAL")
    )
    assert result.record.invalid_action_count == result.record.step_count
    assert result.record.steps[0].action["outcome"] == "unknown_config"


def test_a_failed_remote_inference_does_not_end_the_episode(data, contract) -> None:
    """Losing the link is a condition the policy must handle, not a reason to stop measuring."""
    episode = next(e for e in data.episodes() if e.episode_id == "EPISODE_003")
    result = run_episode(
        data=data, episode=episode, contract=contract, policy=StaticPolicy("CFG_REMOTE_STRONG")
    )

    assert result.record.failed_inference_count > 0
    assert result.record.termination_reason is TerminationReason.PATH_COMPLETE
    assert result.record.final_path_progress == 1.0


def test_a_slow_configuration_skips_frames_and_processes_fewer(data, contract) -> None:
    episode = next(e for e in data.episodes() if e.episode_id == "EPISODE_001")
    fast = run_episode(
        data=data, episode=episode, contract=contract, policy=StaticPolicy("CFG_LOCAL_LIGHT")
    ).record
    slow = run_episode(
        data=data, episode=episode, contract=contract, policy=StaticPolicy("CFG_REMOTE_STRONG")
    ).record

    assert slow.frames_skipped_total > 0
    assert fast.frames_skipped_total == 0
    assert slow.step_count < fast.step_count


def test_switches_are_counted_on_what_actually_ran(data, first_episode, contract) -> None:
    static = run_episode(
        data=data, episode=first_episode, contract=contract, policy=StaticPolicy("CFG_LOCAL_LIGHT")
    ).record
    adaptive = run_episode(
        data=data, episode=first_episode, contract=contract, policy_name="rule_based"
    ).record

    assert static.configuration_switch_count == 0
    assert adaptive.configuration_switch_count > 0


def test_the_adaptation_log_is_recorded_without_being_scored(
    data, first_episode, contract
) -> None:
    """Adaptation latency is not a V1 metric; the data to define it later is kept anyway."""
    record = run_episode(data=data, episode=first_episode, contract=contract).record

    assert record.network_change_times_s == (320.0, 640.0)
    assert len(record.config_selection_history) == record.step_count
    assert record.battery_event_times_s


# --- the runner stays task-agnostic -----------------------------------------------------------


def test_the_runner_imports_no_task_executor_or_policy() -> None:
    """The core architectural rule, checked rather than trusted."""
    import ast
    import pathlib

    from aerointentbench.simulator import episode_runner

    source = pathlib.Path(episode_runner.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = [
        module
        for module in imported
        if module.startswith(("aerointentbench.tasks", "aerointentbench.policies"))
        or module.endswith(("profile_executor", "replay_executor"))
    ]
    assert not forbidden, f"the runner must not import {forbidden}"

    # Attribute access, not a substring search: the module docstring says the runner
    # "never branches on task_id", and searching the text would flag that sentence.
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "task_id" not in attributes, "the runner must not read task identity"


def test_the_executor_is_replaceable_without_touching_the_runner(
    data, first_episode, contract
) -> None:
    """Extensibility criterion 3, exercised through the real runner."""

    class ConstantExecutor:
        def execute(self, request):
            return ExecutionResult(
                success=False,
                latency_s=1.0,
                onboard_energy_j=1.0,
                failure_reason=FailureReason.BACKEND_UNAVAILABLE,
            )

    result = run_episode(
        data=data, episode=first_episode, contract=contract, executor=ConstantExecutor()
    )
    assert result.record.failed_inference_count == result.record.step_count
    assert result.metrics.quality_value == 0.0, "no evidence was ever gathered"
    assert not result.metrics.mission_success


# --- the composition root -------------------------------------------------------------------------


def test_the_suite_aggregates_every_episode(data, contract) -> None:
    result = run_suite(
        data=data, episodes=data.episodes(), contract=contract, policy_name="rule_based"
    )
    assert result.aggregate.episode_count == 3
    assert result.aggregate.mission_success_rate == 1.0


def test_the_rule_based_policy_beats_every_static_baseline(data, contract) -> None:
    """The benchmark's reason to exist: adaptation must be worth something."""
    rates = {
        name: run_suite(
            data=data, episodes=data.episodes(), contract=contract, policy_name=name
        ).aggregate.mission_success_rate
        for name in (
            "always_local_light",
            "always_local_strong",
            "always_remote_strong",
            "rule_based",
        )
    }
    assert rates["rule_based"] > max(
        rate for name, rate in rates.items() if name != "rule_based"
    )


def test_each_baseline_fails_for_its_own_reason(data, contract) -> None:
    """An informative suite: the baselines are not all wrong in the same way."""
    light = run_suite(
        data=data, episodes=data.episodes(), contract=contract, policy_name="always_local_light"
    ).aggregate
    remote = run_suite(
        data=data, episodes=data.episodes(), contract=contract, policy_name="always_remote_strong"
    ).aggregate

    assert light.quality_success_rate == 0.0, "light never reaches the quality threshold"
    assert light.communication_constraint_success_rate == 1.0

    assert remote.quality_success_rate == 1.0, "remote scores highest of all"
    assert remote.communication_constraint_success_rate == 0.0, "and cannot afford to"


def test_hiding_profiles_is_a_run_level_setting(data, contract) -> None:
    disclosed = run_suite(
        data=data, episodes=data.episodes(), contract=contract, policy_name="rule_based"
    ).aggregate
    hidden = run_suite(
        data=data,
        episodes=data.episodes(),
        contract=contract,
        policy_name="rule_based",
        disclose_profiles=False,
    ).aggregate

    assert disclosed.mission_success_rate > hidden.mission_success_rate


def test_a_contract_for_an_unsupported_metric_fails_before_running(
    data, first_episode, contract
) -> None:
    import dataclasses

    from aerointentbench.schemas import SchemaValidationError

    with pytest.raises(SchemaValidationError, match="does not support quality metric"):
        run_episode(
            data=data,
            episode=first_episode,
            contract=dataclasses.replace(contract, quality_metric="mIoU"),
        )


def test_a_missing_data_root_is_reported_clearly(tmp_path: Path) -> None:
    from aerointentbench.schemas import SchemaValidationError

    with pytest.raises(SchemaValidationError, match="benchmark data directory not found"):
        BenchmarkData(tmp_path / "absent")
