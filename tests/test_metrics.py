"""Episode and aggregate metrics, computed from records."""

from __future__ import annotations

import pytest

from aerointentbench.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    compute_episode_metrics,
)
from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.simulator.battery_model import EnergyUsage
from aerointentbench.simulator.records import EpisodeRecord, StepRecord
from aerointentbench.simulator.termination import TerminationReason
from aerointentbench.tasks.base import TaskEvaluationResult


def _evaluation(*, value: float = 0.90, threshold: float = 0.80) -> TaskEvaluationResult:
    return TaskEvaluationResult(
        metric_name="target_f1",
        value=value,
        operator=ComparisonOperator.GREATER_EQUAL,
        threshold=threshold,
        success=value >= threshold,
        details={"matched_targets": 18, "total_targets": 20},
    )


def _step(*, success: bool = True, latency_s: float = 0.45) -> StepRecord:
    return StepRecord(
        step_index=0,
        time_s=0.0,
        frame_id=0,
        state={},
        action={},
        executed_config_id="CFG_LOCAL_STRONG",
        execution={
            "success": success,
            "latency_s": latency_s,
            "failure_reason": None if success else "network_unavailable",
        },
        energy={},
        elapsed_s=1.0,
        frames_skipped=0,
        is_switch=False,
    )


def _record(**overrides) -> EpisodeRecord:
    defaults = dict(
        episode_id="EPISODE_001",
        contract_id="CONTRACT_001",
        policy_name="rule_based",
        executor_id="profile",
        steps=(_step(),),
        termination_reason=TerminationReason.PATH_COMPLETE,
        final_time_s=900.0,
        final_battery_frac=0.33,
        final_path_progress=1.0,
        cumulative_energy=EnergyUsage(flight_j=162_000.0, compute_j=8_105.0, communication_j=190.0),
        cumulative_communication_mb=380.0,
        evidence=None,
        invalid_action_count=0,
        privacy_violation_count=0,
        failed_inference_count=0,
        configuration_switch_count=1,
        frames_skipped_total=0,
    )
    return EpisodeRecord(**{**defaults, **overrides})


# --- constraint evaluation ---------------------------------------------------------------


def test_a_clean_run_succeeds_on_every_constraint(contract) -> None:
    metrics = compute_episode_metrics(_record(), _evaluation(), contract)

    assert metrics.mission_success
    assert metrics.constraint_violation_count == 0
    assert (metrics.deadline_violation_s, metrics.communication_violation_mb) == (0.0, 0.0)


def test_mission_success_needs_every_constraint(contract) -> None:
    """One failure is enough; they are conjunctive, not weighted."""
    over_budget = _record(cumulative_communication_mb=contract.communication_budget_mb + 1.0)
    metrics = compute_episode_metrics(over_budget, _evaluation(), contract)

    assert metrics.quality_success
    assert metrics.deadline_success
    assert not metrics.communication_constraint_success
    assert not metrics.mission_success


@pytest.mark.parametrize(
    ("field", "value", "attribute", "margin_attribute", "expected_margin"),
    [
        ("final_time_s", 970.0, "deadline_success", "deadline_violation_s", 10.0),
        ("final_battery_frac", 0.15, "battery_constraint_success", "battery_violation_frac", 0.05),
        (
            "cumulative_communication_mb",
            425.0,
            "communication_constraint_success",
            "communication_violation_mb",
            25.0,
        ),
    ],
)
def test_each_violation_reports_its_margin(
    contract,
    field: str,
    value: float,
    attribute: str,
    margin_attribute: str,
    expected_margin: float,
) -> None:
    """Missing a deadline by half a second and by five minutes are both False."""
    metrics = compute_episode_metrics(_record(**{field: value}), _evaluation(), contract)
    assert not getattr(metrics, attribute)
    assert getattr(metrics, margin_attribute) == pytest.approx(expected_margin)


def test_a_satisfied_constraint_reports_no_margin(contract) -> None:
    """The margin is the violation, not the headroom, so a pass reports zero."""
    metrics = compute_episode_metrics(_record(final_time_s=100.0), _evaluation(), contract)
    assert metrics.deadline_success
    assert metrics.deadline_violation_s == 0.0


def test_finishing_exactly_on_the_deadline_is_on_time(contract) -> None:
    metrics = compute_episode_metrics(
        _record(final_time_s=contract.deadline_s), _evaluation(), contract
    )
    assert metrics.deadline_success


def test_the_violation_count_is_of_categories_not_time_steps(contract) -> None:
    """A constraint violated for 900 steps is one violated constraint."""
    broken = _record(
        final_time_s=2000.0,
        final_battery_frac=0.05,
        cumulative_communication_mb=9999.0,
        privacy_violation_count=900,
        steps=tuple(_step() for _ in range(900)),
    )
    metrics = compute_episode_metrics(broken, _evaluation(value=0.1), contract)
    assert metrics.constraint_violation_count == 5


def test_privacy_fails_on_an_attempt_not_an_execution(contract) -> None:
    """The validator blocks the execution; choosing a forbidden configuration is the violation."""
    metrics = compute_episode_metrics(_record(privacy_violation_count=3), _evaluation(), contract)
    assert not metrics.privacy_constraint_success
    assert not metrics.mission_success


def test_quality_comes_from_the_task_verdict_not_from_this_layer(contract) -> None:
    metrics = compute_episode_metrics(_record(), _evaluation(value=0.5), contract)
    assert not metrics.quality_success
    assert metrics.quality.metric_name == "target_f1"
    assert metrics.quality_value == 0.5


def test_the_metrics_layer_names_no_task_metric() -> None:
    """Adding a task metric must not require editing episode_metrics.py.

    Checks executable code rather than the file's text: the module's own docstring says
    "nothing here names target_f1", and a substring search would flag that sentence.
    """
    import ast
    import pathlib

    from aerointentbench.metrics import episode_metrics

    tree = ast.parse(pathlib.Path(episode_metrics.__file__).read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    for task_specific in ("target_f1", "target_recall", "target_precision", "mask_iou"):
        assert task_specific not in literals, f"{task_specific} appears as a string literal"
        assert task_specific not in attributes, f"{task_specific} is read as an attribute"


# --- resource metrics ------------------------------------------------------------------------


def test_mean_latency_excludes_failed_executions(contract) -> None:
    """A timeout is a fixed penalty, not a measurement of how long inference takes."""
    record = _record(
        steps=(
            _step(latency_s=0.4),
            _step(latency_s=0.6),
            _step(success=False, latency_s=2.0),
        ),
        failed_inference_count=1,
    )
    metrics = compute_episode_metrics(record, _evaluation(), contract)

    assert metrics.mean_end_to_end_inference_latency_ms == pytest.approx(500.0)
    assert metrics.failed_inference_count == 1


def test_mean_latency_of_no_successful_execution_is_zero(contract) -> None:
    record = _record(steps=(_step(success=False),), failed_inference_count=1)
    assert (
        compute_episode_metrics(
            record, _evaluation(), contract
        ).mean_end_to_end_inference_latency_ms
        == 0.0
    )


def test_energy_components_stay_separable(contract) -> None:
    metrics = compute_episode_metrics(_record(), _evaluation(), contract)
    assert metrics.flight_energy_j == 162_000.0
    assert metrics.compute_energy_j == 8_105.0
    assert metrics.communication_energy_j == 190.0
    assert metrics.total_energy_j == pytest.approx(170_295.0)


def test_metrics_serialise(contract) -> None:
    import json

    payload = compute_episode_metrics(_record(), _evaluation(), contract).to_dict()
    assert json.loads(json.dumps(payload))["mission_success"] is True
    assert set(payload) >= {"quality", "constraints", "violations", "resources", "behaviour"}


# --- aggregation ------------------------------------------------------------------------------


def _metrics(contract, **overrides) -> EpisodeMetrics:
    return compute_episode_metrics(_record(**overrides), _evaluation(), contract)


def test_mission_success_rate_is_the_fraction_that_passed(contract) -> None:
    passing = _metrics(contract)
    failing = _metrics(contract, cumulative_communication_mb=9999.0)

    assert aggregate_metrics([passing, passing, failing]).mission_success_rate == pytest.approx(
        2 / 3
    )


def test_the_rates_decompose_the_headline(contract) -> None:
    """When success is low, the rates say which constraint kept breaking."""
    aggregate = aggregate_metrics(
        [_metrics(contract), _metrics(contract, cumulative_communication_mb=9999.0)]
    )
    assert aggregate.mission_success_rate == 0.5
    assert aggregate.quality_success_rate == 1.0
    assert aggregate.communication_constraint_success_rate == 0.5


def test_means_are_over_episodes_not_steps(contract) -> None:
    """A long episode must not dominate the average latency."""
    short = _metrics(contract, steps=(_step(latency_s=0.1),))
    long = _metrics(contract, steps=tuple(_step(latency_s=0.9) for _ in range(500)))

    aggregate = aggregate_metrics([short, long])
    assert aggregate.mean_end_to_end_inference_latency_ms == pytest.approx(500.0)


def test_totals_are_summed_not_averaged(contract) -> None:
    aggregate = aggregate_metrics(
        [_metrics(contract, invalid_action_count=2), _metrics(contract, invalid_action_count=3)]
    )
    assert aggregate.total_invalid_action_count == 5


def test_an_empty_suite_reports_zeros_rather_than_raising(contract) -> None:
    """Reporting "no episodes ran" beats an exception halfway through a campaign."""
    aggregate = aggregate_metrics([])
    assert isinstance(aggregate, AggregateMetrics)
    assert aggregate.episode_count == 0
    assert aggregate.mission_success_rate == 0.0


def test_aggregate_serialises(contract) -> None:
    import json

    payload = aggregate_metrics([_metrics(contract)]).to_dict()
    assert json.loads(json.dumps(payload))["mission_success_rate"] == 1.0
    assert set(payload) >= {"episode_count", "mission_success_rate", "rates", "means", "totals"}


def test_aggregation_is_order_independent(contract) -> None:
    a = _metrics(contract)
    b = _metrics(contract, cumulative_communication_mb=9999.0)
    assert aggregate_metrics([a, b]).to_dict() == aggregate_metrics([b, a]).to_dict()


def test_a_new_metric_needs_only_the_record(contract) -> None:
    """Records carry more than V1 scores, so a later metric needs no rerun."""
    record = _record(
        steps=tuple(_step() for _ in range(5)),
        network_change_times_s=(320.0, 640.0),
        battery_event_times_s=(100.0,),
        config_selection_history=("CFG_LOCAL_LIGHT",) * 5,
    )
    metrics = compute_episode_metrics(record, _evaluation(), contract)
    assert metrics.step_count == 5
    # Nothing in V1 consumes these, and that is the point of logging them.
    assert record.network_change_times_s == (320.0, 640.0)
    assert record.to_dict()["adaptation_log"]["battery_event_times_s"] == [100.0]


def test_the_default_record_serialisation_withholds_the_answers() -> None:
    """A shared result file must not publish ground-truth-derived fields."""
    import json

    from aerointentbench.tasks.human_search_segmentation import PredictedInstance
    from aerointentbench.tasks.human_search_segmentation.evidence_tracker import (
        HumanSearchEvidenceRecord,
    )

    evidence = HumanSearchEvidenceRecord(
        processed_frames=1,
        instances=(
            PredictedInstance(
                prediction_id="P0",
                frame_id=0,
                predicted_target_id="PT_1",
                confidence=0.9,
                ground_truth_track_id="GT_SECRET",
                mask_iou=0.77,
            ),
        ),
    )
    record = _record(evidence=evidence)

    default = json.dumps(record.to_dict())
    detailed = json.dumps(record.to_dict(include_detail=True))

    assert "GT_SECRET" not in default
    assert "mask_iou" not in default
    assert default.count("processed_frames") == 1, "the summary is still reported"
    assert "GT_SECRET" in detailed


# --- uncertainty ------------------------------------------------------------------------


def test_wilson_interval_brackets_the_estimate() -> None:
    from aerointentbench.metrics.aggregate_metrics import wilson_interval

    for successes, trials in ((0, 3), (2, 3), (3, 3), (138, 150), (144, 150)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= successes / trials <= high <= 1.0


def test_the_interval_is_defined_at_the_boundaries() -> None:
    """A policy that failed everything still has a real upper bound worth reporting."""
    from aerointentbench.metrics.aggregate_metrics import wilson_interval

    low, high = wilson_interval(0, 150)
    assert low == 0.0
    assert 0.0 < high < 0.05

    low, high = wilson_interval(150, 150)
    assert high == 1.0
    assert 0.95 < low < 1.0


def test_the_interval_narrows_with_the_sample() -> None:
    """Three episodes cannot resolve what a hundred and fifty can."""
    from aerointentbench.metrics.aggregate_metrics import wilson_interval

    def width(successes: int, trials: int) -> float:
        low, high = wilson_interval(successes, trials)
        return high - low

    assert width(2, 3) > 0.7, "two of three is almost no information"
    assert width(100, 150) < 0.15


def test_an_empty_suite_has_a_degenerate_interval() -> None:
    from aerointentbench.metrics.aggregate_metrics import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_aggregate_reports_the_count_and_the_interval(contract) -> None:
    passing = _metrics(contract)
    failing = _metrics(contract, cumulative_communication_mb=9999.0)

    aggregate = aggregate_metrics([passing, passing, failing])

    assert aggregate.mission_success_count == 2
    assert aggregate.episode_count == 3
    low, high = aggregate.mission_success_ci
    assert low < aggregate.mission_success_rate < high
    assert aggregate.to_dict()["mission_success_count"] == 2
    assert aggregate.to_dict()["mission_success_ci_95"] == [low, high]
