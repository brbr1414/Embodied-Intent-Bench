"""Empirical mask replay: masks, IoU, one-to-one matching, and unit-consistent scoring.

Two metric families are kept strictly apart and never divided into each other:

- **Mission, track-level**: ``target_recall`` = unique ground-truth tracks found / total valid
  tracks. This is the canonical empirical mission-quality metric.
- **Frame, detection-level**: ``detection_precision`` = matched predictions / non-ignored
  predictions, plus the false-positive burden (`false_positive_detections`,
  `false_positives_per_minute`).

Track-level precision and F1 are unavailable without persistent predicted-track identity, so
they are surfaced as ``None`` with a reason, never as a mixed-unit number. Everything is
checked against hand-calculated values on the tiny fixture under
``data/examples/empirical_replay`` -- a correctness example, not a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerointentbench.benchmark import BenchmarkData, run_episode
from aerointentbench.policies.static import StaticPolicy
from aerointentbench.schemas import load_contract, load_episode, load_task_spec
from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.tasks.human_search_segmentation import (
    EmpiricalQualityScores,
    FrameGroundTruth,
    HumanSearchEvidenceRecord,
    HumanSearchMaskGroundTruth,
    HumanSearchSegmentationEvaluator,
    MaskPredictedInstance,
    MaskTarget,
    decode_mask,
    load_any_ground_truth,
    load_human_search_mask_ground_truth,
    mask_iou,
    match_frame,
)
from aerointentbench.tasks.human_search_segmentation.masks import BinaryMask

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data" / "examples" / "empirical_replay"
EXAMPLE_EPISODE = EXAMPLE_ROOT / "episodes" / "episode.json"
EXAMPLE_CONTRACT = EXAMPLE_ROOT / "contracts" / "contract.json"
EXAMPLE_TASK_SPEC = EXAMPLE_ROOT / "task_specs" / "human_search.json"
EXAMPLE_GROUND_TRUTH = EXAMPLE_ROOT / "ground_truth" / "gt_example_stream.json"


# --- helpers ---------------------------------------------------------------------------------


def mask(*rows: str) -> BinaryMask:
    """Build a mask from a bitmap: one string per row, '1' set, '0' clear."""
    return decode_mask({"height": len(rows), "width": len(rows[0]), "rows": list(rows)})


def prediction(
    prediction_id: str,
    binary_mask: BinaryMask,
    *,
    frame_id: int = 0,
    category: str = "person",
    confidence: float = 0.9,
) -> MaskPredictedInstance:
    return MaskPredictedInstance(
        prediction_id=prediction_id,
        frame_id=frame_id,
        predicted_target_id=prediction_id,
        confidence=confidence,
        category=category,
        mask=binary_mask,
    )


def target(
    track_id: str, binary_mask: BinaryMask, *, category: str = "person", ignore: bool = False
) -> MaskTarget:
    return MaskTarget(track_id=track_id, category=category, mask=binary_mask, ignore=ignore)


def record(*instances: MaskPredictedInstance, frames: int = 1) -> HumanSearchEvidenceRecord:
    return HumanSearchEvidenceRecord(processed_frames=frames, instances=instances)


def stream(frames: dict[int, list[MaskTarget]]) -> HumanSearchMaskGroundTruth:
    return HumanSearchMaskGroundTruth(
        frame_stream_id="S",
        frames=tuple(FrameGroundTruth(fid, tuple(insts)) for fid, insts in sorted(frames.items())),
    )


@pytest.fixture
def rule():
    return load_task_spec(EXAMPLE_TASK_SPEC).matching_rule


@pytest.fixture
def evaluator() -> HumanSearchSegmentationEvaluator:
    return HumanSearchSegmentationEvaluator(load_task_spec(EXAMPLE_TASK_SPEC))


BOX = ("1100", "1100", "0000", "0000")  # 4 px, upper-left
FAR = ("0000", "0000", "0011", "0011")  # 4 px, lower-right, disjoint from BOX


# --- mask IoU --------------------------------------------------------------------------------


def test_identical_masks_have_iou_one() -> None:
    a = mask("0110", "0110", "0000", "0000")
    assert mask_iou(a, a) == 1.0


def test_disjoint_masks_have_iou_zero() -> None:
    assert mask_iou(mask(*BOX), mask(*FAR)) == 0.0


def test_partial_overlap_is_the_hand_calculated_value() -> None:
    # a: rows 0-1 cols 0-1; b: rows 0-1 cols 1-2. Shared column is 2 px, union 6 px -> 1/3.
    a = mask("1100", "1100", "0000", "0000")
    b = mask("0110", "0110", "0000", "0000")
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_masks_of_different_dimensions_do_not_compare_silently() -> None:
    with pytest.raises(SchemaValidationError, match="different dimensions"):
        mask_iou(mask("11", "11"), mask("111", "111"))


def test_both_empty_masks_score_zero_not_a_match() -> None:
    empty = mask("0000", "0000")
    assert mask_iou(empty, empty) == 0.0


def test_empty_against_non_empty_is_zero() -> None:
    assert mask_iou(mask("0000", "0000"), mask("1100", "0000")) == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {"height": 2, "width": 2, "rows": ["10", "1"]},  # ragged row
        {"height": 3, "width": 2, "rows": ["10", "01"]},  # row count != height
        {"height": 2, "width": 2, "rows": ["1x", "01"]},  # illegal character
        {"height": 0, "width": 2, "rows": []},  # non-positive dimension
        {"height": 2, "width": 2},  # missing rows
        {"height": 2, "width": 2, "rows": ["10", "01"], "extra": 1},  # unknown field
        ["not", "an", "object"],  # not a mapping
    ],
)
def test_malformed_masks_fail_validation(payload: object) -> None:
    with pytest.raises(SchemaValidationError):
        decode_mask(payload)


# --- one-to-one matching ---------------------------------------------------------------------


def test_one_prediction_cannot_match_two_targets(rule) -> None:
    box = mask(*BOX)
    matching = match_frame(
        0, [prediction("P", box)], [target("A", box), target("B", box)], rule=rule
    )
    assert len(matching.matches) == 1
    assert len(matching.matched_track_ids) == 1


def test_two_predictions_cannot_both_match_one_target(rule) -> None:
    box = mask(*BOX)
    matching = match_frame(
        0, [prediction("P1", box), prediction("P2", box)], [target("A", box)], rule=rule
    )
    assert len(matching.matches) == 1
    assert matching.unmatched_prediction_ids == ("P2",)  # P1 wins on the tie-break


def test_a_prediction_below_threshold_stays_unmatched(rule) -> None:
    gt = mask("1111", "1111", "1111", "1111")  # 16 px
    weak = mask("1100", "1100", "0000", "0000")  # 4 px inside -> IoU 0.25
    matching = match_frame(0, [prediction("P", weak)], [target("A", gt)], rule=rule)
    assert matching.matches == ()
    assert matching.unmatched_prediction_ids == ("P",)


def test_matching_is_deterministic_under_ties(rule) -> None:
    box = mask(*BOX)
    predictions = [prediction("P_b", box), prediction("P_a", box)]
    first = match_frame(0, predictions, [target("A", box)], rule=rule)
    second = match_frame(0, list(reversed(predictions)), [target("A", box)], rule=rule)
    assert first.matches[0].prediction_id == "P_a"  # lower id wins, order-independent
    assert second.matches[0].prediction_id == "P_a"


# --- mission target recall: track-level on both sides (points 1, 2) --------------------------


def test_target_recall_divides_unique_tracks_by_unique_tracks(evaluator) -> None:
    a, b, c = mask(*BOX), mask(*FAR), mask("0011", "0011", "0000", "0000")
    gt = stream({0: [target("A", a), target("B", b), target("C", c)]})
    scores = evaluator.score_empirical(record(prediction("PA", a), prediction("PB", b)), gt)
    assert isinstance(scores, EmpiricalQualityScores)
    assert scores.unique_targets_found == 2
    assert scores.total_unique_targets == 3
    assert scores.target_recall == pytest.approx(2 / 3)


def test_one_track_seen_in_many_frames_does_not_increase_recall_twice(evaluator) -> None:
    box = mask(*BOX)
    gt = stream({0: [target("A", box)], 1: [target("A", box)], 2: [target("A", box)]})
    evidence = record(
        prediction("P0", box, frame_id=0),
        prediction("P1", box, frame_id=1),
        prediction("P2", box, frame_id=2),
        frames=3,
    )
    scores = evaluator.score_empirical(evidence, gt)
    assert scores.unique_targets_found == 1  # three sightings, one track
    assert scores.total_unique_targets == 1
    assert scores.target_recall == 1.0
    # ...but the detections are three, and all matched.
    assert scores.matched_detections == 3


def test_a_missed_track_costs_recall(evaluator) -> None:
    a, b = mask(*BOX), mask(*FAR)
    gt = stream({0: [target("A", a), target("B", b)]})
    scores = evaluator.score_empirical(record(prediction("PA", a)), gt)  # B never predicted
    assert scores.unique_targets_found == 1
    assert scores.total_unique_targets == 2
    assert scores.target_recall == 0.5


# --- detection precision: frame-level on both sides (points 3, 4) ----------------------------


def test_detection_precision_divides_matched_detections_by_total_predictions(evaluator) -> None:
    box = mask(*BOX)
    gt = stream({0: [target("A", box)], 1: [target("A", box)]})
    # Frame 0: one match + one false positive. Frame 1: one match. matched 2, total 3.
    evidence = record(
        prediction("P0match", box, frame_id=0),
        prediction("P0fp", mask(*FAR), frame_id=0),
        prediction("P1match", box, frame_id=1),
        frames=2,
    )
    scores = evaluator.score_empirical(evidence, gt)
    assert scores.matched_detections == 2
    assert scores.total_predictions == 3
    assert scores.false_positive_detections == 1
    assert scores.detection_precision == pytest.approx(2 / 3)
    assert scores.unique_targets_found == 1  # detection precision is not track recall


def test_repeated_false_positives_lower_detection_precision_consistently(evaluator) -> None:
    box = mask(*BOX)
    gt = stream({0: [target("A", box)]})
    # One true match, three false positives spread over frames -> precision 1/4.
    evidence = record(
        prediction("Pmatch", box, frame_id=0),
        prediction("Pfp0", mask(*FAR), frame_id=0),
        prediction("Pfp1", mask(*FAR), frame_id=1),
        prediction("Pfp2", mask(*FAR), frame_id=2),
        frames=3,
    )
    scores = evaluator.score_empirical(evidence, gt)
    assert scores.false_positive_detections == 3
    assert scores.total_predictions == 4
    assert scores.detection_precision == 0.25


# --- false-positive burden (point 5) ---------------------------------------------------------


def test_false_positives_per_minute_is_computed_from_processed_frames(evaluator) -> None:
    # 2 false positives over 6 processed frames == 2 per 6 s == 20 per minute (1 fps nominal).
    gt = HumanSearchMaskGroundTruth(frame_stream_id="S", frames=())
    evidence = record(
        prediction("FP0", mask(*FAR), frame_id=0),
        prediction("FP1", mask(*FAR), frame_id=1),
        frames=6,
    )
    scores = evaluator.score_empirical(evidence, gt)
    assert scores.false_positive_detections == 2
    assert scores.false_positives_per_minute == pytest.approx(20.0)


def test_false_positives_per_minute_of_nothing_examined_is_zero(evaluator) -> None:
    empty_gt = HumanSearchMaskGroundTruth(frame_stream_id="S", frames=())
    assert evaluator.score_empirical(record(frames=0), empty_gt).false_positives_per_minute == 0.0


# --- ignore regions (point 6) ----------------------------------------------------------------


def test_ignored_region_predictions_are_not_false_positives(evaluator) -> None:
    box = mask("1111", "1111", "1111", "1111")
    gt = stream({0: [target("IGNORE", box, ignore=True)]})
    # A prediction on the ignore region is neither matched nor a false positive, and the
    # region is not a target to miss: everything is vacuously perfect.
    scores = evaluator.score_empirical(record(prediction("P", box)), gt)
    assert scores.total_unique_targets == 0
    assert scores.false_positive_detections == 0
    assert scores.total_predictions == 0
    assert scores.target_recall == 1.0
    assert scores.detection_precision == 1.0


def test_a_wrong_category_prediction_is_neither_matched_nor_a_false_positive(evaluator) -> None:
    box = mask(*BOX)
    gt = stream({0: [target("A", box)]})
    scores = evaluator.score_empirical(record(prediction("V", box, category="vehicle")), gt)
    assert scores.matched_detections == 0
    assert scores.false_positive_detections == 0
    assert scores.total_predictions == 0
    assert scores.total_unique_targets == 1


# --- track-level F1 is unavailable, not mixed (point 7) --------------------------------------


def test_empirical_scores_report_no_track_level_f1(evaluator) -> None:
    box = mask(*BOX)
    gt = stream({0: [target("A", box)]})
    scores = evaluator.score_empirical(record(prediction("PA", box)), gt)
    details = scores.to_dict()
    assert details["target_f1"] is None
    assert details["track_level_precision"] is None
    assert details["track_level_metrics_available"] is False
    assert (
        "reason" in details["track_level_unavailable_reason"]
        or details["track_level_unavailable_reason"]
    )
    # No mixed-unit precision under a track-level name at all.
    assert "target_precision" not in details


def test_empirical_value_of_refuses_track_level_metrics(evaluator) -> None:
    box = mask(*BOX)
    scores = evaluator.score_empirical(
        record(prediction("PA", box)), stream({0: [target("A", box)]})
    )
    assert scores.value_of("target_recall") == 1.0
    assert scores.value_of("detection_precision") == 1.0
    for unavailable in ("target_precision", "target_f1"):
        with pytest.raises(SchemaValidationError, match="does not support quality metric"):
            scores.value_of(unavailable)


def test_score_refuses_mask_ground_truth(evaluator) -> None:
    box = mask(*BOX)
    with pytest.raises(SchemaValidationError, match="score_empirical"):
        evaluator.score(record(prediction("PA", box)), stream({0: [target("A", box)]}))


# --- the mask ground-truth loader ------------------------------------------------------------


def test_the_example_mask_ground_truth_loads() -> None:
    gt = load_human_search_mask_ground_truth(EXAMPLE_GROUND_TRUTH)
    assert isinstance(gt, HumanSearchMaskGroundTruth)
    assert gt.frame_stream_id == "EXAMPLE_STREAM"
    assert gt.target_track_ids() == frozenset({"GT_A", "GT_B", "GT_C"})  # GT_IGNORE excluded


def test_the_dispatcher_picks_the_mask_loader_by_the_frames_field() -> None:
    assert isinstance(load_any_ground_truth(EXAMPLE_GROUND_TRUTH), HumanSearchMaskGroundTruth)


def test_ground_truth_declaring_neither_form_is_rejected(write_json) -> None:
    payload = {
        "schema_version": "1.0",
        "frame_stream_id": "S",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
    }
    with pytest.raises(SchemaValidationError, match="exactly one of"):
        load_any_ground_truth(write_json(payload))


def test_a_prediction_may_not_carry_a_ground_truth_track_id() -> None:
    payload = {
        "prediction_id": "P",
        "confidence": 0.9,
        "ground_truth_track_id": "GT_A",
        "mask": {"height": 2, "width": 2, "rows": ["11", "00"]},
    }
    with pytest.raises(SchemaValidationError, match="ground_truth_track_id"):
        MaskPredictedInstance.from_dict(payload)


# --- replay executor returns masks, provenance, missing records ------------------------------


def test_replay_executor_returns_the_stored_masks_and_metadata() -> None:
    from aerointentbench.executor import ExecutionRequest
    from aerointentbench.executor.replay_executor import make_replay_executor

    data = BenchmarkData(EXAMPLE_ROOT)
    executor = make_replay_executor(EXAMPLE_ROOT / "predictions" / "empirical_replay.json")
    request = ExecutionRequest(
        episode_id="EMPIRICAL_EXAMPLE_EPISODE",
        frame_id=0,
        configuration=data.catalog.get("CFG_LOCAL_STRONG"),
        network=None,
        current_time_s=0.0,
        seed=7,
    )
    result = executor.execute(request)
    assert result.success
    assert result.latency_s == pytest.approx(0.120)
    assert result.onboard_energy_j == 5.0
    prediction_ids = {inst["prediction_id"] for inst in result.prediction["instances"]}
    assert prediction_ids == {"P0a", "P0b", "P0c"}
    assert "rows" in result.prediction["instances"][0]["mask"]


def test_a_missing_replay_record_fails_clearly_in_strict_mode() -> None:
    data = BenchmarkData(EXAMPLE_ROOT)
    with pytest.raises(SchemaValidationError, match="replay gap"):
        run_episode(
            data=data,
            episode=load_episode(EXAMPLE_EPISODE),
            contract=load_contract(EXAMPLE_CONTRACT),
            policy=StaticPolicy("CFG_LOCAL_STRONG"),
            executor_name="replay",
            replay_strict=True,
        )


def test_a_stray_scalar_mask_iou_on_the_wire_is_not_trusted(evaluator) -> None:
    gt_box = mask("1111", "1111", "1111", "1111")
    weak = {
        "prediction_id": "P",
        "confidence": 0.9,
        "mask_iou": 0.99,  # a lie; the mask says 0.25
        "mask": {"height": 4, "width": 4, "rows": ["1100", "1100", "0000", "0000"]},
    }
    instance = MaskPredictedInstance.from_dict(weak, default_frame_id=0)
    scores = evaluator.score_empirical(record(instance), stream({0: [target("A", gt_box)]}))
    assert scores.unique_targets_found == 0  # scored on the mask, not the scalar


# --- policy boundary (point 9) ---------------------------------------------------------------


def _run_example(policy=None, executor_name="replay"):
    return run_episode(
        data=BenchmarkData(EXAMPLE_ROOT),
        episode=load_episode(EXAMPLE_EPISODE),
        contract=load_contract(EXAMPLE_CONTRACT),
        policy=policy if policy is not None else StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name=executor_name,
    )


def test_runtime_state_carries_no_ground_truth_or_evaluator_metric() -> None:
    result = _run_example()
    forbidden = (
        "mask",
        "track_id",
        "ground_truth",
        "recall",
        "true_positive",
        "detection_precision",
        "GT_",
    )
    for step in result.record.steps:
        blob = json.dumps(step.state)
        for token in forbidden:
            assert token not in blob, f"{token!r} leaked into policy-visible state"
        assert set(step.state["evidence_summary"]) == {
            "predicted_unique_targets",
            "processed_frames",
            "mean_prediction_confidence",
        }


# --- end to end: target_recall contract, provenance, two executors (points 8, 10, 11) --------


def test_a_target_recall_contract_is_evaluated_on_recall() -> None:
    result = _run_example(policy=StaticPolicy("CFG_LOCAL_STRONG"))
    quality = result.metrics.quality
    assert quality.metric_name == "target_recall"
    assert quality.value == pytest.approx(2 / 3)
    assert quality.threshold == 0.50
    assert quality.success is True  # 2/3 >= 0.50


def test_the_empirical_example_runs_through_the_cli_with_verified_values(tmp_path: Path) -> None:
    from aerointentbench.run_benchmark import main

    output = tmp_path / "result.json"
    exit_code = main(
        [
            "--data-root",
            str(EXAMPLE_ROOT),
            "--episode",
            str(EXAMPLE_EPISODE),
            "--contract",
            str(EXAMPLE_CONTRACT),
            "--policy",
            "rule_based",
            "--executor",
            "replay",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["executor_id"] == "replay"

    quality = payload["episodes"][0]["quality"]
    assert quality["metric_name"] == "target_recall"
    assert quality["value"] == pytest.approx(2 / 3)

    d = quality["details"]
    assert d["quality_evaluation"] == "empirical_mask_iou"
    # mission, track-level
    assert d["unique_targets_found"] == 2
    assert d["total_unique_targets"] == 3
    assert d["target_recall"] == pytest.approx(2 / 3)
    # frame, detection-level
    assert d["matched_detections"] == 3
    assert d["total_predictions"] == 5
    assert d["false_positive_detections"] == 2
    assert d["detection_precision"] == pytest.approx(0.6)
    assert d["false_positives_per_minute"] == pytest.approx(40.0)  # 2 fp over 3 frames
    # explicitly unavailable
    assert d["target_f1"] is None
    assert d["track_level_metrics_available"] is False


def test_no_empirical_metric_mixes_unique_tracks_with_frame_false_positives(evaluator) -> None:
    """The regression this branch fixes: precision must not be TP_tracks / (TP_tracks + FP_dets)."""
    a, b, c = mask(*BOX), mask(*FAR), mask("0011", "0011", "0000", "0000")
    gt = stream({0: [target("A", a), target("B", b), target("C", c)]})
    # 2 matched tracks, 1 false positive detection.
    scores = evaluator.score_empirical(
        record(
            prediction("PA", a),
            prediction("PB", b),
            prediction("G", mask("0001", "0000", "1000", "0000")),
        ),
        gt,
    )
    # The old mixed value was 2/(2+1) = 0.667. The correct detection precision is
    # matched_detections/total = 2/3 too here, but by construction of DETECTIONS, not tracks:
    assert scores.matched_detections == 2 and scores.total_predictions == 3
    assert scores.detection_precision == pytest.approx(2 / 3)
    # recall is a pure track ratio, independent of the false positive.
    assert scores.target_recall == pytest.approx(2 / 3)
    # and there is simply no track-level precision/F1 number to be mixed.
    assert scores.to_dict()["target_f1"] is None


def test_profile_mode_on_a_mask_stream_still_runs_and_is_scored_empirically() -> None:
    # Profile has no synthetic mask generator for a mask stream, so it gathers no evidence:
    # recall 0 over three targets. Same runner, still routed to empirical scoring by the GT.
    profile = _run_example(executor_name="profile")
    replay = _run_example(executor_name="replay")
    assert profile.metrics.quality.details["target_recall"] == 0.0
    assert profile.metrics.executor_id == "profile"
    assert replay.metrics.quality.details["unique_targets_found"] == 2
    assert replay.metrics.executor_id == "replay"
    assert replay.metrics.quality.details["quality_evaluation"] == "empirical_mask_iou"
