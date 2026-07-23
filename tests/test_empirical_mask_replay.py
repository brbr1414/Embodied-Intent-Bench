"""Empirical mask replay: masks, IoU, one-to-one matching, track-level scoring, and the
end-to-end path through the runner, replay executor, and CLI.

Where the precomputed path trusts a scalar ``mask_iou``, this path computes IoU from real
prediction and ground-truth masks and matches one-to-one before deduplicating finds by hidden
track ID. Everything is checked against hand-calculated numbers on the tiny fixture under
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
    predicted_target_id: str | None = None,
) -> MaskPredictedInstance:
    return MaskPredictedInstance(
        prediction_id=prediction_id,
        frame_id=frame_id,
        predicted_target_id=predicted_target_id or prediction_id,
        confidence=confidence,
        category=category,
        mask=binary_mask,
    )


def target(
    track_id: str, binary_mask: BinaryMask, *, category: str = "person", ignore: bool = False
) -> MaskTarget:
    return MaskTarget(track_id=track_id, category=category, mask=binary_mask, ignore=ignore)


@pytest.fixture
def rule():
    return load_task_spec(EXAMPLE_TASK_SPEC).matching_rule


@pytest.fixture
def evaluator() -> HumanSearchSegmentationEvaluator:
    return HumanSearchSegmentationEvaluator(load_task_spec(EXAMPLE_TASK_SPEC))


def record(*instances: MaskPredictedInstance, frames: int = 1) -> HumanSearchEvidenceRecord:
    return HumanSearchEvidenceRecord(processed_frames=frames, instances=instances)


# --- mask IoU --------------------------------------------------------------------------------


def test_identical_masks_have_iou_one() -> None:
    a = mask("0110", "0110", "0000", "0000")
    assert mask_iou(a, a) == 1.0


def test_disjoint_masks_have_iou_zero() -> None:
    a = mask("1100", "1100", "0000", "0000")
    b = mask("0000", "0000", "0011", "0011")
    assert mask_iou(a, b) == 0.0


def test_partial_overlap_is_the_hand_calculated_value() -> None:
    # a: 4 pixels (rows 0-1, cols 0-1); b: 4 pixels (rows 0-1, cols 1-2).
    # intersection is the shared column (2 px); union is 6 px -> 1/3.
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
    box = mask("1100", "1100", "0000", "0000")
    matching = match_frame(
        0, [prediction("P", box)], [target("A", box), target("B", box)], rule=rule
    )
    assert len(matching.matches) == 1
    assert len(matching.matched_track_ids) == 1


def test_two_predictions_cannot_both_match_one_target(rule) -> None:
    box = mask("1100", "1100", "0000", "0000")
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
    box = mask("1100", "1100", "0000", "0000")
    predictions = [prediction("P_b", box), prediction("P_a", box)]
    first = match_frame(0, predictions, [target("A", box)], rule=rule)
    second = match_frame(0, list(reversed(predictions)), [target("A", box)], rule=rule)
    # The lower prediction_id wins whichever order they arrive in.
    assert first.matches[0].prediction_id == "P_a"
    assert second.matches[0].prediction_id == "P_a"


def test_category_filtering_excludes_the_wrong_category(evaluator) -> None:
    box = mask("1111", "1111", "1111", "1111")
    gt = HumanSearchMaskGroundTruth(
        frame_stream_id="S", frames=(FrameGroundTruth(0, (target("A", box),)),)
    )
    # A vehicle prediction sitting exactly on the person target neither finds it nor is a
    # false positive: it is not of the target category, so scoring never considers it.
    evidence = record(prediction("V", box, category="vehicle"))
    scores = evaluator.score(evidence, gt)
    assert scores.matched_targets == 0
    assert scores.false_positive_targets == 0
    assert scores.total_targets == 1


def test_ignored_targets_absorb_predictions_and_leave_the_metric_unchanged(evaluator) -> None:
    box = mask("1111", "1111", "1111", "1111")
    gt = HumanSearchMaskGroundTruth(
        frame_stream_id="S",
        frames=(FrameGroundTruth(0, (target("IGNORE", box, ignore=True),)),),
    )
    # A prediction on an ignore region is neither a find nor a false positive, and the region
    # is not a target to miss: everything is vacuously perfect.
    scores = evaluator.score(record(prediction("P", box)), gt)
    assert scores.total_targets == 0
    assert scores.false_positive_targets == 0
    assert (scores.target_precision, scores.target_recall) == (1.0, 1.0)


# --- track-level mission scoring -------------------------------------------------------------


def _stream(frames: dict[int, list[MaskTarget]]) -> HumanSearchMaskGroundTruth:
    return HumanSearchMaskGroundTruth(
        frame_stream_id="S",
        frames=tuple(FrameGroundTruth(fid, tuple(insts)) for fid, insts in sorted(frames.items())),
    )


def test_one_track_seen_in_many_frames_is_one_found_target(evaluator) -> None:
    box = mask("1100", "1100", "0000", "0000")
    gt = _stream({0: [target("A", box)], 1: [target("A", box)], 2: [target("A", box)]})
    evidence = record(
        prediction("P0", box, frame_id=0),
        prediction("P1", box, frame_id=1),
        prediction("P2", box, frame_id=2),
        frames=3,
    )
    scores = evaluator.score(evidence, gt)
    assert scores.matched_targets == 1
    assert scores.total_targets == 1
    assert (scores.target_recall, scores.target_precision) == (1.0, 1.0)


def test_two_distinct_tracks_count_as_two_found_targets(evaluator) -> None:
    a = mask("1100", "1100", "0000", "0000")
    b = mask("0000", "0000", "0011", "0011")
    gt = _stream({0: [target("A", a), target("B", b)]})
    scores = evaluator.score(record(prediction("PA", a), prediction("PB", b)), gt)
    assert scores.matched_targets == 2
    assert scores.target_recall == 1.0


def test_a_false_positive_costs_precision(evaluator) -> None:
    a = mask("1100", "1100", "0000", "0000")
    ghost = mask("0000", "0000", "0011", "0011")
    gt = _stream({0: [target("A", a)]})
    scores = evaluator.score(record(prediction("PA", a), prediction("GHOST", ghost)), gt)
    assert scores.matched_targets == 1
    assert scores.false_positive_targets == 1
    assert scores.target_precision == 0.5
    assert scores.target_recall == 1.0


def test_a_missed_track_costs_recall(evaluator) -> None:
    a = mask("1100", "1100", "0000", "0000")
    b = mask("0000", "0000", "0011", "0011")
    gt = _stream({0: [target("A", a), target("B", b)]})
    scores = evaluator.score(record(prediction("PA", a)), gt)  # B never predicted
    assert scores.matched_targets == 1
    assert scores.total_targets == 2
    assert scores.target_recall == 0.5
    assert scores.target_precision == 1.0


def test_mission_f1_matches_the_hand_calculation(evaluator) -> None:
    # tp = 2, fp = 1, total = 3 -> precision 2/3, recall 2/3, f1 2/3.
    a = mask("1100", "1100", "0000", "0000")
    b = mask("0000", "0000", "0011", "0011")
    c = mask("0011", "0011", "0000", "0000")
    gt = _stream({0: [target("A", a), target("B", b), target("C", c)]})
    ghost = mask("0000", "1000", "0001", "0000")
    scores = evaluator.score(
        record(prediction("PA", a), prediction("PB", b), prediction("G", ghost)), gt
    )
    assert (scores.matched_targets, scores.false_positive_targets, scores.total_targets) == (
        2,
        1,
        3,
    )
    assert scores.target_precision == pytest.approx(2 / 3)
    assert scores.target_recall == pytest.approx(2 / 3)
    assert scores.target_f1 == pytest.approx(2 / 3)


# --- the mask ground-truth loader ------------------------------------------------------------


def test_the_example_mask_ground_truth_loads() -> None:
    gt = load_human_search_mask_ground_truth(EXAMPLE_GROUND_TRUTH)
    assert isinstance(gt, HumanSearchMaskGroundTruth)
    assert gt.frame_stream_id == "EXAMPLE_STREAM"
    # GT_A, GT_B, GT_C are scored; GT_IGNORE is not counted.
    assert gt.target_track_ids() == frozenset({"GT_A", "GT_B", "GT_C"})


def test_the_dispatcher_picks_the_mask_loader_by_the_frames_field() -> None:
    gt = load_any_ground_truth(EXAMPLE_GROUND_TRUTH)
    assert isinstance(gt, HumanSearchMaskGroundTruth)


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


# --- policy boundary: nothing ground-truth-derived reaches the policy ------------------------


def _run_example(policy=None, executor_name="replay"):
    data = BenchmarkData(EXAMPLE_ROOT)
    episode = load_episode(EXAMPLE_EPISODE)
    contract = load_contract(EXAMPLE_CONTRACT)
    return run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy=policy if policy is not None else StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name=executor_name,
    )


def test_runtime_state_carries_no_ground_truth() -> None:
    result = _run_example()
    forbidden = ("mask", "track_id", "ground_truth", "recall", "true_positive", "GT_")
    for step in result.record.steps:
        blob = json.dumps(step.state)
        for token in forbidden:
            assert token not in blob, f"{token!r} leaked into policy-visible state"
        # The evidence summary a policy sees is prediction-derived counts only.
        assert set(step.state["evidence_summary"]) == {
            "predicted_unique_targets",
            "processed_frames",
            "mean_prediction_confidence",
        }


# --- the replay executor returns masks, and provenance is honest -----------------------------


def test_replay_executor_returns_the_stored_masks_and_metadata() -> None:
    from aerointentbench.executor import ExecutionRequest
    from aerointentbench.executor.replay_executor import make_replay_executor

    data = BenchmarkData(EXAMPLE_ROOT)
    executor = make_replay_executor(EXAMPLE_ROOT / "predictions" / "empirical_replay.json")
    request = ExecutionRequest(
        episode_id="EMPIRICAL_EXAMPLE_EPISODE",
        frame_id=0,
        configuration=data.catalog.get("CFG_LOCAL_STRONG"),
        network=None,  # unused by replay
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


def test_empirical_replay_uses_recorded_masks_not_synthetic_generation() -> None:
    # Profile mode has no synthetic mask generator for a mask stream, so it gathers no
    # evidence and finds nothing; replay serves the recorded masks and finds two of three.
    profile = _run_example(executor_name="profile")
    replay = _run_example(executor_name="replay")
    assert profile.metrics.quality.details["target_recall"] == 0.0
    assert replay.metrics.quality.details["matched_targets"] == 2
    assert replay.metrics.quality.value == pytest.approx(4 / 7)  # f1 of (0.5, 2/3)


def test_a_stray_scalar_mask_iou_on_the_wire_is_not_trusted(rule, evaluator) -> None:
    # The mask says IoU 0.25 (below threshold); a lying mask_iou=0.99 must not change that.
    gt_box = mask("1111", "1111", "1111", "1111")
    weak = {
        "prediction_id": "P",
        "confidence": 0.9,
        "mask_iou": 0.99,
        "mask": {"height": 4, "width": 4, "rows": ["1100", "1100", "0000", "0000"]},
    }
    instance = MaskPredictedInstance.from_dict(weak, default_frame_id=0)
    gt = _stream({0: [target("A", gt_box)]})
    scores = evaluator.score(record(instance), gt)
    assert scores.matched_targets == 0  # scored on the mask, not the scalar


def test_a_missing_replay_record_fails_clearly_in_strict_mode() -> None:
    data = BenchmarkData(EXAMPLE_ROOT)
    episode = load_episode(EXAMPLE_EPISODE)
    contract = load_contract(EXAMPLE_CONTRACT)
    with pytest.raises(SchemaValidationError, match="replay gap"):
        run_episode(
            data=data,
            episode=episode,
            contract=contract,
            policy=StaticPolicy("CFG_LOCAL_STRONG"),
            executor_name="replay",
            replay_strict=True,
        )


# --- end to end: CLI, provenance, and the same runner over two executors ---------------------


def test_the_empirical_example_runs_through_the_cli(tmp_path: Path) -> None:
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

    details = payload["episodes"][0]["quality"]["details"]
    assert details["quality_evaluation"] == "empirical_mask_iou"
    assert details["matched_targets"] == 2
    assert details["total_targets"] == 3
    assert details["false_positive_targets"] == 2
    assert details["target_precision"] == pytest.approx(0.5)
    assert details["target_recall"] == pytest.approx(2 / 3)
    assert details["target_f1"] == pytest.approx(4 / 7)


def test_empirical_provenance_distinguishes_it_from_profile() -> None:
    # Same episode, same policy interface, two executors, one unmodified EpisodeRunner.
    profile = _run_example(executor_name="profile")
    replay = _run_example(executor_name="replay")
    assert profile.metrics.executor_id == "profile"
    assert replay.metrics.executor_id == "replay"
    # The mask-scoring tag marks the empirical result; profile+mask-GT is scored the same way
    # but produced no masked evidence.
    assert replay.metrics.quality.details["quality_evaluation"] == "empirical_mask_iou"


def test_the_hand_calculated_metrics_hold_under_a_static_policy() -> None:
    result = _run_example(policy=StaticPolicy("CFG_LOCAL_STRONG"))
    details = result.metrics.quality.details
    assert (
        details["matched_targets"],
        details["total_targets"],
        details["false_positive_targets"],
    ) == (
        2,
        3,
        2,
    )
    assert result.metrics.quality.value == pytest.approx(4 / 7)
