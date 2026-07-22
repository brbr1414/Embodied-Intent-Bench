"""The human-search task: ground truth, prediction synthesis, evidence, and scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from aerointentbench.executor import ExecutionRequest, ExecutionResult, FailureReason
from aerointentbench.schemas import NetworkObservation, QualityTier, SchemaValidationError
from aerointentbench.tasks.base import TaskEvaluationResult
from aerointentbench.tasks.human_search_segmentation import (
    FramePrediction,
    HumanSearchEvidenceTracker,
    HumanSearchSegmentationEvaluator,
    HumanSearchSegmentationTask,
    PredictedInstance,
    SyntheticHumanSearchPredictions,
    TargetTrack,
    load_human_search_ground_truth,
)
from aerointentbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceRecord,
)
from aerointentbench.tasks.human_search_segmentation.ground_truth import HumanSearchGroundTruth
from aerointentbench.tasks.registry import resolve_task, task_registry

NETWORK = NetworkObservation(bandwidth_mbps=20.0, rtt_ms=30.0, packet_loss_frac=0.0)


@pytest.fixture
def ground_truth(data_dir: Path) -> HumanSearchGroundTruth:
    return load_human_search_ground_truth(
        data_dir / "ground_truth" / "synthetic_human_search_stream_001.json"
    )


def _instance(
    *,
    frame_id: int = 0,
    predicted_target_id: str = "PT_1",
    gt: str | None = "GT_1",
    iou: float = 0.9,
    confidence: float = 0.8,
) -> PredictedInstance:
    return PredictedInstance(
        prediction_id=f"P{frame_id}_{predicted_target_id}",
        frame_id=frame_id,
        predicted_target_id=predicted_target_id,
        confidence=confidence,
        ground_truth_track_id=gt,
        mask_iou=iou,
    )


def _success(*instances: PredictedInstance, frame_id: int = 0) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        latency_s=0.1,
        onboard_energy_j=1.0,
        prediction=FramePrediction(frame_id=frame_id, instances=instances),
    )


# --- ground truth --------------------------------------------------------------------


def test_ground_truth_fixture_loads(ground_truth: HumanSearchGroundTruth) -> None:
    assert ground_truth.frame_stream_id == "STREAM_001"
    assert ground_truth.task_id == "HUMAN_SEARCH_SEGMENTATION"
    assert len(ground_truth) == 20


def test_visibility_is_an_inclusive_interval() -> None:
    target = TargetTrack(track_id="GT_1", first_frame_id=10, last_frame_id=12)
    assert [target.is_visible_at(frame) for frame in (9, 10, 11, 12, 13)] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert target.visible_frame_count == 3


def test_visible_at_selects_the_targets_in_view(ground_truth: HumanSearchGroundTruth) -> None:
    assert [target.track_id for target in ground_truth.visible_at(110)] == ["GT_F01"]
    assert list(ground_truth.visible_at(5)) == []


def test_the_fixture_contains_fleeting_targets(ground_truth: HumanSearchGroundTruth) -> None:
    """Short visibility is the only thing that makes latency cost quality.

    A target in view for one frame is missed outright by a configuration slow enough to skip
    it. Without such targets, every configuration finds everything and the benchmark
    measures nothing.
    """
    shortest = min(target.visible_frame_count for target in ground_truth.targets)
    fleeting = [t for t in ground_truth.targets if t.visible_frame_count <= 2]
    assert shortest == 1
    assert len(fleeting) >= 8, "too few fleeting targets for latency to matter"


def test_duplicate_track_ids_are_rejected(write_json) -> None:
    target = {"track_id": "GT_1", "first_frame_id": 0, "last_frame_id": 1}
    payload = {
        "schema_version": "1.0",
        "frame_stream_id": "S",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
        "targets": [target, dict(target)],
    }
    with pytest.raises(SchemaValidationError, match="duplicate target track_id"):
        load_human_search_ground_truth(write_json(payload))


def test_an_inverted_interval_is_rejected(write_json) -> None:
    payload = {
        "schema_version": "1.0",
        "frame_stream_id": "S",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
        "targets": [{"track_id": "GT_1", "first_frame_id": 10, "last_frame_id": 5}],
    }
    with pytest.raises(SchemaValidationError, match="must be >= 10"):
        load_human_search_ground_truth(write_json(payload))


def test_ground_truth_for_another_task_is_rejected(write_json) -> None:
    payload = {
        "schema_version": "1.0",
        "frame_stream_id": "S",
        "task_id": "OBJECT_DETECTION",
        "targets": [{"track_id": "GT_1", "first_frame_id": 0, "last_frame_id": 1}],
    }
    with pytest.raises(SchemaValidationError, match="but the file declares"):
        load_human_search_ground_truth(write_json(payload))


# --- synthetic predictions -------------------------------------------------------------


@pytest.fixture
def predictions(ground_truth, profiles) -> SyntheticHumanSearchPredictions:
    return SyntheticHumanSearchPredictions(ground_truth, profiles)


def _request(catalog, config_id: str, frame_id: int, seed: int = 42) -> ExecutionRequest:
    return ExecutionRequest(
        episode_id="EPISODE_001",
        frame_id=frame_id,
        configuration=catalog.get(config_id),
        network=NETWORK,
        current_time_s=float(frame_id),
        seed=seed,
    )


def test_prediction_generation_is_reproducible(predictions, catalog) -> None:
    """Builtin hash() is per-process salted; using it would make episodes irreproducible."""
    request = _request(catalog, "CFG_LOCAL_STRONG", 50)
    first = predictions.prediction_for(request)
    second = predictions.prediction_for(request)
    assert first == second


def test_predictions_appear_only_where_a_target_is_visible(predictions, catalog) -> None:
    empty = predictions.prediction_for(_request(catalog, "CFG_REMOTE_STRONG", 5))
    assert [i for i in empty.instances if i.ground_truth_track_id is not None] == []


def test_a_better_tier_finds_more_of_what_is_there(predictions, catalog, ground_truth) -> None:
    """The tier is the whole quality signal, and it must actually move the outcome."""
    found: dict[str, int] = {}
    for config_id in ("CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG", "CFG_REMOTE_STRONG"):
        detected = set()
        for target in ground_truth.targets:
            for frame in range(target.first_frame_id, target.last_frame_id + 1):
                for instance in predictions.prediction_for(
                    _request(catalog, config_id, frame)
                ).instances:
                    if instance.ground_truth_track_id is not None:
                        detected.add(instance.ground_truth_track_id)
        found[config_id] = len(detected)

    assert found["CFG_LOCAL_LIGHT"] < found["CFG_REMOTE_STRONG"]


def test_a_weak_tier_can_see_a_target_without_segmenting_it_well_enough(
    predictions, catalog, ground_truth
) -> None:
    """Detection and matching are separate; the IoU range straddles the matching threshold."""
    below_threshold = 0
    for target in ground_truth.targets[:6]:
        for frame in range(target.first_frame_id, target.last_frame_id + 1):
            for instance in predictions.prediction_for(
                _request(catalog, "CFG_LOCAL_LIGHT", frame)
            ).instances:
                if instance.ground_truth_track_id is not None and instance.mask_iou < 0.50:
                    below_threshold += 1
    assert below_threshold > 0


def test_a_predicted_identity_is_stable_across_frames(predictions, catalog, ground_truth) -> None:
    """A tracker re-identifies a person, so repeated sightings are one predicted target."""
    target = next(t for t in ground_truth.targets if t.visible_frame_count > 20)
    identities = set()
    for frame in range(target.first_frame_id, target.last_frame_id + 1):
        for instance in predictions.prediction_for(
            _request(catalog, "CFG_REMOTE_STRONG", frame)
        ).instances:
            if instance.ground_truth_track_id == target.track_id:
                identities.add(instance.predicted_target_id)
    assert len(identities) == 1


def test_the_predicted_identity_is_not_the_ground_truth_track(
    predictions, catalog, ground_truth
) -> None:
    track_ids = ground_truth.track_ids
    for frame in (50, 130, 700):
        for instance in predictions.prediction_for(
            _request(catalog, "CFG_REMOTE_STRONG", frame)
        ).instances:
            assert instance.predicted_target_id not in track_ids


# --- evidence tracker --------------------------------------------------------------------


def test_tracker_counts_processed_frames() -> None:
    tracker = HumanSearchEvidenceTracker()
    for frame in range(3):
        tracker.update(_success(frame_id=frame))
    assert tracker.policy_summary().processed_frames == 3


def test_a_failed_execution_is_not_a_processed_frame() -> None:
    """It cost time and energy, but the system never saw anything."""
    tracker = HumanSearchEvidenceTracker()
    tracker.update(_success(_instance()))
    tracker.update(
        ExecutionResult(
            success=False,
            latency_s=2.0,
            onboard_energy_j=1.0,
            failure_reason=FailureReason.NETWORK_UNAVAILABLE,
        )
    )
    assert tracker.policy_summary().processed_frames == 1


def test_the_policy_sees_its_own_identities_not_confirmed_finds() -> None:
    """A false positive inflates the count, and the policy cannot tell."""
    tracker = HumanSearchEvidenceTracker()
    tracker.update(_success(_instance(predicted_target_id="PT_1", gt="GT_1")))
    tracker.update(_success(_instance(predicted_target_id="PT_GHOST", gt=None, iou=0.0)))

    assert tracker.policy_summary().predicted_unique_targets == 2


def test_repeated_sightings_of_one_identity_count_once() -> None:
    tracker = HumanSearchEvidenceTracker()
    for frame in range(5):
        tracker.update(_success(_instance(frame_id=frame, predicted_target_id="PT_1")))
    assert tracker.policy_summary().predicted_unique_targets == 1


def test_the_two_deduplications_are_deliberately_different() -> None:
    """One predicted identity covering two real people is one to the policy, two to the evaluator.

    Collapsing the two would hand the policy its own true positive count.
    """
    tracker = HumanSearchEvidenceTracker()
    tracker.update(_success(_instance(predicted_target_id="PT_1", gt="GT_1")))
    tracker.update(_success(_instance(predicted_target_id="PT_1", gt="GT_2", frame_id=1)))

    assert tracker.policy_summary().predicted_unique_targets == 1
    matched = {i.ground_truth_track_id for i in tracker.final_record().instances}
    assert matched == {"GT_1", "GT_2"}


def test_mean_confidence_is_over_instances() -> None:
    tracker = HumanSearchEvidenceTracker()
    tracker.update(_success(_instance(confidence=0.6), _instance(predicted_target_id="PT_2", confidence=0.8)))
    assert tracker.policy_summary().mean_prediction_confidence == pytest.approx(0.7)


def test_mean_confidence_of_nothing_is_zero_not_an_error() -> None:
    assert HumanSearchEvidenceTracker().policy_summary().mean_prediction_confidence == 0.0


def test_the_tracker_rejects_a_payload_from_another_task() -> None:
    tracker = HumanSearchEvidenceTracker()
    result = ExecutionResult(
        success=True, latency_s=0.1, onboard_energy_j=1.0, prediction={"boxes": []}
    )
    with pytest.raises(TypeError, match="expects a FramePrediction"):
        tracker.update(result)


# --- evaluator ------------------------------------------------------------------------------


@pytest.fixture
def evaluator(task_spec_fixture) -> HumanSearchSegmentationEvaluator:
    return HumanSearchSegmentationEvaluator(task_spec_fixture)


@pytest.fixture
def task_spec_fixture(data_dir: Path):
    from aerointentbench.schemas import load_task_spec

    return load_task_spec(data_dir / "task_specs" / "human_search_segmentation.json")


def _gt(*track_ids: str) -> HumanSearchGroundTruth:
    return HumanSearchGroundTruth(
        frame_stream_id="S",
        targets=tuple(
            TargetTrack(track_id=track_id, first_frame_id=0, last_frame_id=10)
            for track_id in track_ids
        ),
    )


def _record(*instances: PredictedInstance, frames: int = 1) -> HumanSearchEvidenceRecord:
    return HumanSearchEvidenceRecord(processed_frames=frames, instances=instances)


def test_a_perfect_run_scores_one(evaluator) -> None:
    record = _record(
        _instance(predicted_target_id="PT_1", gt="GT_1"),
        _instance(predicted_target_id="PT_2", gt="GT_2"),
    )
    scores = evaluator.score(record, _gt("GT_1", "GT_2"))
    assert (scores.target_recall, scores.target_precision, scores.target_f1) == (1.0, 1.0, 1.0)


def test_a_missed_target_costs_recall_only(evaluator) -> None:
    scores = evaluator.score(
        _record(_instance(predicted_target_id="PT_1", gt="GT_1")), _gt("GT_1", "GT_2")
    )
    assert scores.target_recall == 0.5
    assert scores.target_precision == 1.0
    assert scores.target_f1 == pytest.approx(2 / 3)


def test_a_false_positive_costs_precision_only(evaluator) -> None:
    scores = evaluator.score(
        _record(
            _instance(predicted_target_id="PT_1", gt="GT_1"),
            _instance(predicted_target_id="PT_GHOST", gt=None, iou=0.0),
        ),
        _gt("GT_1"),
    )
    assert scores.target_recall == 1.0
    assert scores.target_precision == 0.5
    assert scores.false_positive_targets == 1


@pytest.mark.parametrize(
    ("iou", "counts"), [(0.49, False), (0.50, True), (0.51, True)]
)
def test_matching_applies_the_task_rule_at_its_threshold(evaluator, iou: float, counts: bool) -> None:
    """Seeing a person is not the same as segmenting them well enough to count."""
    scores = evaluator.score(
        _record(_instance(predicted_target_id="PT_1", gt="GT_1", iou=iou)), _gt("GT_1")
    )
    assert (scores.matched_targets == 1) is counts


def test_a_detection_that_fails_matching_is_a_false_positive(evaluator) -> None:
    scores = evaluator.score(
        _record(_instance(predicted_target_id="PT_1", gt="GT_1", iou=0.2)), _gt("GT_1")
    )
    assert scores.target_recall == 0.0
    assert scores.target_precision == 0.0
    assert scores.predicted_targets == 1


def test_sixty_sightings_of_one_person_are_one_find(evaluator) -> None:
    record = _record(
        *(
            _instance(frame_id=frame, predicted_target_id="PT_1", gt="GT_1")
            for frame in range(60)
        ),
        frames=60,
    )
    scores = evaluator.score(record, _gt("GT_1"))
    assert scores.matched_targets == 1
    assert scores.target_recall == 1.0
    assert scores.target_precision == 1.0


def test_an_episode_that_predicted_nothing_scores_zero_recall(evaluator) -> None:
    scores = evaluator.score(_record(frames=0), _gt("GT_1", "GT_2"))
    assert scores.target_recall == 0.0
    assert scores.target_precision == 1.0, "predicting nothing produces no false positives"
    assert scores.target_f1 == 0.0


def test_a_scene_with_no_targets_is_vacuously_perfect(evaluator) -> None:
    scores = evaluator.score(_record(frames=1), _gt())
    assert (scores.target_recall, scores.target_precision, scores.target_f1) == (1.0, 1.0, 1.0)


def test_evaluation_applies_the_contract_operator_and_threshold(evaluator, contract) -> None:
    record = _record(
        _instance(predicted_target_id="PT_1", gt="GT_1"),
        _instance(predicted_target_id="PT_2", gt="GT_2"),
    )
    result = evaluator.evaluate(record, _gt("GT_1", "GT_2"), contract)

    assert isinstance(result, TaskEvaluationResult)
    assert result.metric_name == "target_f1"
    assert result.value == 1.0
    assert result.threshold == contract.quality_threshold
    assert result.success is True
    assert result.details["matched_targets"] == 2


def test_the_result_reports_every_metric_even_though_one_decides(evaluator, contract) -> None:
    result = evaluator.evaluate(
        _record(_instance(predicted_target_id="PT_1", gt="GT_1")), _gt("GT_1", "GT_2"), contract
    )
    assert set(result.details) >= {"target_precision", "target_recall", "target_f1"}
    assert result.metric_name == "target_f1"


def test_an_unsupported_metric_is_refused(evaluator, contract) -> None:
    import dataclasses

    with pytest.raises(SchemaValidationError, match="cannot compute quality metric"):
        evaluator.evaluate(
            _record(), _gt("GT_1"), dataclasses.replace(contract, quality_metric="mIoU")
        )


# --- the task registry --------------------------------------------------------------------------


def test_the_task_resolves_from_its_specification(task_spec_fixture) -> None:
    task = resolve_task(task_spec_fixture)
    assert isinstance(task, HumanSearchSegmentationTask)
    assert task.task_id == "HUMAN_SEARCH_SEGMENTATION"


def test_each_episode_gets_a_fresh_tracker(task_spec_fixture) -> None:
    """Trackers are stateful; sharing one would carry evidence between episodes."""
    task = resolve_task(task_spec_fixture)
    first, second = task.create_tracker(), task.create_tracker()
    first.update(_success(_instance()))
    assert second.policy_summary().processed_frames == 0


def test_a_mismatched_specification_is_refused(task_spec_fixture) -> None:
    import dataclasses

    with pytest.raises(SchemaValidationError, match="implements"):
        HumanSearchSegmentationTask(dataclasses.replace(task_spec_fixture, task_id="OTHER"))


def test_a_new_task_needs_no_change_to_anything_else(task_spec_fixture) -> None:
    """Extensibility criterion 1: adding a task evaluator is a registration, not an edit."""
    from aerointentbench.registry import Registry
    from aerointentbench.tasks.base import TaskDefinition

    class DetectionTask:
        task_id = "OBJECT_DETECTION"
        task_spec = task_spec_fixture

        def create_tracker(self):
            return HumanSearchEvidenceTracker()

        def create_evaluator(self):
            return HumanSearchSegmentationEvaluator(task_spec_fixture)

    scratch: Registry[TaskDefinition] = Registry("task")
    scratch.register("OBJECT_DETECTION", lambda **_: DetectionTask())
    assert scratch.create("OBJECT_DETECTION").task_id == "OBJECT_DETECTION"
    assert "OBJECT_DETECTION" not in task_registry, "the scratch registry must stay isolated"
