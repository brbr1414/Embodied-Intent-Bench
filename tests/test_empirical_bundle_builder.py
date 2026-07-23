"""The empirical-bundle builder: manifest parsing, source conversion, validation, and the
full path from source files through the builder and validator to a benchmark run.

The builder executes no model and measures no hardware; it converts externally produced
files (ground-truth masks, per-config predictions, measured metadata) into a self-contained
replay bundle the ordinary CLI runs unchanged. Every expected metric is hand-calculable on
tiny masks. The committed source fixture under ``data/examples/empirical_source`` is
synthetic and only tests conversion correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerointentbench.benchmark import BenchmarkData, run_episode
from aerointentbench.policies.static import StaticPolicy
from aerointentbench.schemas.contract import load_contract
from aerointentbench.schemas.episode import load_episode
from aerointentbench.schemas.loading import SchemaValidationError, SchemaVersionError
from aerointentbench.tasks.human_search_segmentation import load_any_ground_truth
from aerointentbench.tools.build_empirical_bundle import build_empirical_bundle
from aerointentbench.tools.bundle_manifest import (
    load_manifest,
    parse_measurements_source,
    parse_predictions_source,
)
from aerointentbench.tools.validate_empirical_bundle import validate_empirical_bundle

EXAMPLE_SOURCE = Path(__file__).resolve().parents[1] / "data" / "examples" / "empirical_source"
EXAMPLE_MANIFEST = EXAMPLE_SOURCE / "manifest.json"

MASK2 = {"height": 2, "width": 2, "rows": ["11", "00"]}
MASK2_OTHER = {"height": 2, "width": 2, "rows": ["00", "11"]}


# --- a minimal, mutable source bundle for unit and negative tests ----------------------------


def make_source(
    root: Path,
    *,
    manifest_extra: dict | None = None,
    frames: list[dict] | None = None,
    gt_instances: list[dict] | None = None,
    predictions: list[dict] | None = None,
    measurement_rows: list[str] | None = None,
    config_id: str = "CFG_LOCAL_LIGHT",
) -> Path:
    """Write a minimal valid source set into ``root`` and return the manifest path."""
    root.mkdir(parents=True, exist_ok=True)
    frames = frames if frames is not None else [{"frame_id": 0, "timestamp_s": 0.0}]
    gt_instances = (
        gt_instances
        if gt_instances is not None
        else [{"frame_id": 0, "track_id": "GT_X", "category": "person", "mask": MASK2}]
    )
    predictions = (
        predictions
        if predictions is not None
        else [
            {
                "frame_id": 0,
                "prediction_id": "P0",
                "category": "person",
                "confidence": 0.9,
                "mask": MASK2,
            }
        ]
    )
    measurement_rows = (
        measurement_rows if measurement_rows is not None else ["0,true,0.10,1.0,0.0,0.0,"]
    )

    (root / "ground_truth.json").write_text(
        json.dumps({"schema_version": "1.0", "frame_stream_id": "S", "instances": gt_instances}),
        encoding="utf-8",
    )
    (root / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8"
    )
    (root / "measurements.csv").write_text(
        "frame_id,success,latency_s,compute_energy_j,upload_mb,download_mb,failure_reason\n"
        + "\n".join(measurement_rows)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "bundle_id": "MINI",
        "frame_stream_id": "S",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
        "image_width": 2,
        "image_height": 2,
        "data_origin": "hand_authored_fixture",
        "coverage": "strict",
        "frames": frames,
        "ground_truth_source": "ground_truth.json",
        "configurations": [
            {
                "config_id": config_id,
                "model_id": "M",
                "placement": "local",
                "precision": "int8",
                "quality_tier": "low",
                "prediction_source": "predictions.jsonl",
                "measurement_source": "measurements.csv",
                "prediction_provenance": "hand_authored",
                "measurement_provenance": "estimated",
            }
        ],
        "contract": {
            "quality_metric": "target_recall",
            "quality_threshold": 0.5,
            "deadline_s": 1.0,
        },
        **(manifest_extra or {}),
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def build_mini(tmp_path: Path, **kwargs):
    manifest = make_source(tmp_path / "src", **kwargs)
    return build_empirical_bundle(manifest, tmp_path / "bundle")


# --- manifest and input parsing (points 1-6) -------------------------------------------------


def test_a_valid_manifest_loads() -> None:
    manifest = load_manifest(EXAMPLE_MANIFEST)
    assert manifest.bundle_id == "EMPIRICAL_SOURCE_EXAMPLE"
    assert manifest.frame_ids == (0, 1, 2)
    assert [c.config_id for c in manifest.configurations] == ["CFG_LOCAL_STRONG", "CFG_LOCAL_LIGHT"]


def test_an_unsupported_schema_version_fails(tmp_path: Path) -> None:
    manifest = make_source(tmp_path, manifest_extra={"schema_version": "9.9"})
    with pytest.raises(SchemaVersionError):
        load_manifest(manifest)


def test_duplicate_frame_ids_fail(tmp_path: Path) -> None:
    manifest = make_source(tmp_path, frames=[{"frame_id": 0}, {"frame_id": 0}])
    with pytest.raises(SchemaValidationError, match="duplicate frame_id"):
        load_manifest(manifest)


def test_a_measurement_for_an_unknown_config_frame_fails(tmp_path: Path) -> None:
    # A measurement on a frame the manifest does not declare is rejected during build.
    with pytest.raises(SchemaValidationError, match="not declared in the manifest frames"):
        build_mini(
            tmp_path, measurement_rows=["0,true,0.1,1.0,0.0,0.0,", "5,true,0.1,1.0,0.0,0.0,"]
        )


def test_unsafe_paths_fail(tmp_path: Path) -> None:
    manifest = make_source(tmp_path, manifest_extra={})
    payload = json.loads(manifest.read_text())
    payload["configurations"][0]["prediction_source"] = "../../etc/passwd"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="escapes the manifest directory"):
        load_manifest(manifest)


def test_a_missing_source_file_fails_clearly(tmp_path: Path) -> None:
    manifest = make_source(tmp_path)
    (tmp_path / "predictions.jsonl").unlink()
    with pytest.raises(SchemaValidationError, match="cannot be read"):
        build_empirical_bundle(manifest, tmp_path / "bundle")


# --- mask and prediction conversion (points 7-12) --------------------------------------------


def test_valid_gt_and_prediction_masks_convert(tmp_path: Path) -> None:
    summary = build_mini(tmp_path)
    gt = load_any_ground_truth(tmp_path / "bundle" / "ground_truth" / "ground_truth.json")
    assert gt.target_track_ids() == frozenset({"GT_X"})
    replay = json.loads((tmp_path / "bundle" / "predictions" / "replay.json").read_text())
    instances = replay["records"][0]["prediction"]["instances"]
    assert instances[0]["mask"]["rows"] == MASK2["rows"]
    assert summary.gt_instance_count == 1


def test_a_prediction_with_wrong_dimensions_fails(tmp_path: Path) -> None:
    bad = {
        "frame_id": 0,
        "prediction_id": "P0",
        "category": "person",
        "confidence": 0.9,
        "mask": {"height": 3, "width": 3, "rows": ["000", "010", "000"]},
    }
    with pytest.raises(SchemaValidationError, match="frame is 2x2"):
        build_mini(tmp_path, predictions=[bad])


def test_predictions_containing_gt_track_ids_are_rejected(tmp_path: Path) -> None:
    bad = {
        "frame_id": 0,
        "prediction_id": "P0",
        "category": "person",
        "confidence": 0.9,
        "mask": MASK2,
        "ground_truth_track_id": "GT_X",
    }
    with pytest.raises(SchemaValidationError, match="ground_truth_track_id"):
        build_mini(tmp_path, predictions=[bad])


def test_a_scalar_mask_iou_in_a_prediction_is_rejected(tmp_path: Path) -> None:
    bad = {
        "frame_id": 0,
        "prediction_id": "P0",
        "category": "person",
        "confidence": 0.9,
        "mask": MASK2,
        "mask_iou": 0.99,
    }
    with pytest.raises(SchemaValidationError, match="mask_iou"):
        build_mini(tmp_path, predictions=[bad])


def test_prediction_ordering_is_deterministic(tmp_path: Path) -> None:
    preds = [
        {
            "frame_id": 0,
            "prediction_id": "Pb",
            "category": "person",
            "confidence": 0.5,
            "mask": MASK2,
        },
        {
            "frame_id": 0,
            "prediction_id": "Pa",
            "category": "person",
            "confidence": 0.5,
            "mask": MASK2_OTHER,
        },
    ]
    build_mini(tmp_path, predictions=preds)
    replay = json.loads((tmp_path / "bundle" / "predictions" / "replay.json").read_text())
    ids = [i["prediction_id"] for i in replay["records"][0]["prediction"]["instances"]]
    assert ids == ["Pa", "Pb"]  # sorted by prediction_id


# --- measurement conversion (points 13-18) ---------------------------------------------------


def test_latency_and_energy_are_preserved(tmp_path: Path) -> None:
    build_mini(tmp_path, measurement_rows=["0,true,0.25,7.5,0.0,0.0,"])
    replay = json.loads((tmp_path / "bundle" / "predictions" / "replay.json").read_text())
    record = replay["records"][0]
    assert record["latency_ms"] == pytest.approx(250.0)
    assert record["onboard_energy_j"] == 7.5


def test_negative_latency_fails(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="latency_s"):
        build_mini(tmp_path, measurement_rows=["0,true,-0.1,1.0,0.0,0.0,"])


def test_negative_energy_fails(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="compute_energy_j"):
        build_mini(tmp_path, measurement_rows=["0,true,0.1,-1.0,0.0,0.0,"])


def test_a_blank_required_measurement_is_never_inferred(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="never inferred"):
        build_mini(tmp_path, measurement_rows=["0,true,,1.0,0.0,0.0,"])


def test_missing_measurement_fails_in_strict_mode(tmp_path: Path) -> None:
    # Two frames declared, only one measured -> strict coverage rejects the gap.
    with pytest.raises(SchemaValidationError, match="strict coverage"):
        build_mini(
            tmp_path,
            frames=[{"frame_id": 0}, {"frame_id": 1}],
            gt_instances=[{"frame_id": 0, "track_id": "GT_X", "category": "person", "mask": MASK2}],
            measurement_rows=["0,true,0.1,1.0,0.0,0.0,"],
        )


def test_sparse_mode_turns_a_missing_pair_into_a_failed_record(tmp_path: Path) -> None:
    manifest = make_source(
        tmp_path / "src",
        frames=[{"frame_id": 0}, {"frame_id": 1}],
        gt_instances=[{"frame_id": 0, "track_id": "GT_X", "category": "person", "mask": MASK2}],
        measurement_rows=["0,true,0.1,1.0,0.0,0.0,"],
    )
    build_empirical_bundle(manifest, tmp_path / "bundle", coverage="sparse")
    replay = json.loads((tmp_path / "bundle" / "predictions" / "replay.json").read_text())
    by_frame = {r["frame_id"]: r for r in replay["records"]}
    assert by_frame[1]["success"] is False
    assert by_frame[1]["failure_reason"] == "no_prediction_available"


def test_failed_execution_records_preserve_failure_reasons(tmp_path: Path) -> None:
    build_mini(tmp_path, measurement_rows=["0,false,0.5,1.0,0.0,0.0,network_unavailable"])
    replay = json.loads((tmp_path / "bundle" / "predictions" / "replay.json").read_text())
    record = replay["records"][0]
    assert record["success"] is False
    assert record["failure_reason"] == "network_unavailable"
    assert "prediction" not in record


def test_an_unknown_failure_reason_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="not a known"):
        build_mini(tmp_path, measurement_rows=["0,false,0.5,1.0,0.0,0.0,exploded"])


def test_measurement_provenance_is_preserved(tmp_path: Path) -> None:
    build_mini(tmp_path)
    provenance = json.loads((tmp_path / "bundle" / "provenance.json").read_text())
    assert provenance["data_origin"] == "hand_authored_fixture"
    assert provenance["configurations"][0]["measurement_provenance"] == "estimated"
    assert provenance["configurations"][0]["prediction_provenance"] == "hand_authored"


# --- bundle generation (points 19-24) --------------------------------------------------------


def test_generated_bundle_matches_the_replay_and_gt_schemas(tmp_path: Path) -> None:
    from aerointentbench.executor.replay_executor import load_replay_record_set

    build_mini(tmp_path)
    # Both load through the existing strict loaders unchanged.
    record_set = load_replay_record_set(tmp_path / "bundle" / "predictions" / "replay.json")
    assert record_set.episode_id == "MINI_EPISODE"
    load_any_ground_truth(tmp_path / "bundle" / "ground_truth" / "ground_truth.json")


def test_generated_bundle_is_self_contained(tmp_path: Path) -> None:
    import shutil

    manifest = make_source(tmp_path / "src")
    build_empirical_bundle(manifest, tmp_path / "bundle")
    shutil.rmtree(tmp_path / "src")  # delete the sources entirely
    # The bundle still runs: it copied everything it needs.
    result = _run_bundle(tmp_path / "bundle", fallback="CFG_LOCAL_LIGHT")
    assert result.metrics.quality.details["quality_evaluation"] == "empirical_mask_iou"


def test_provenance_counts_are_correct(tmp_path: Path) -> None:
    summary = build_empirical_bundle(make_source(tmp_path / "src"), tmp_path / "bundle")
    provenance = json.loads((tmp_path / "bundle" / "provenance.json").read_text())
    assert provenance["frame_count"] == 1
    assert provenance["replay_record_count"] == summary.replay_record_count == 1
    assert provenance["ground_truth"]["unique_track_count"] == 1


def test_hashes_are_stable_across_builds(tmp_path: Path) -> None:
    manifest = make_source(tmp_path / "src")
    build_empirical_bundle(manifest, tmp_path / "b1", created_at="FIXED")
    build_empirical_bundle(manifest, tmp_path / "b2", created_at="FIXED")
    p1 = json.loads((tmp_path / "b1" / "provenance.json").read_text())
    p2 = json.loads((tmp_path / "b2" / "provenance.json").read_text())
    assert p1["generated_file_sha256"] == p2["generated_file_sha256"]


def test_building_twice_is_byte_identical_except_the_timestamp(tmp_path: Path) -> None:
    manifest = make_source(tmp_path / "src")
    build_empirical_bundle(manifest, tmp_path / "b1", created_at="FIXED")
    build_empirical_bundle(manifest, tmp_path / "b2", created_at="FIXED")
    for path in (tmp_path / "b1").rglob("*"):
        if path.is_file():
            twin = tmp_path / "b2" / path.relative_to(tmp_path / "b1")
            assert path.read_bytes() == twin.read_bytes(), path


def test_the_existing_hand_authored_empirical_fixture_still_runs() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "examples" / "empirical_replay"
    result = _run_bundle(root, fallback="CFG_LOCAL_LIGHT")
    assert result.metrics.quality.details["unique_targets_found"] == 2


# --- end to end (points 25-29) ---------------------------------------------------------------


def _run_bundle(bundle_root: Path, *, policy="CFG_LOCAL_STRONG", fallback="CFG_LOCAL_LIGHT"):
    data = BenchmarkData(bundle_root)
    episode = load_episode(next((bundle_root / "episodes").glob("*.json")))
    contract = load_contract(next((bundle_root / "contracts").glob("*.json")))
    return run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy=StaticPolicy(policy),
        executor_name="replay",
        fallback_config_id=fallback,
    )


@pytest.fixture
def example_bundle(tmp_path: Path) -> Path:
    build_empirical_bundle(EXAMPLE_MANIFEST, tmp_path / "bundle", created_at="FIXED", validate=True)
    return tmp_path / "bundle"


def test_the_generated_example_bundle_passes_validation(example_bundle: Path) -> None:
    report = validate_empirical_bundle(example_bundle)
    assert report.ok, report.errors
    assert len(report.checks) >= 5


def test_the_generated_example_bundle_produces_expected_metrics(example_bundle: Path) -> None:
    result = _run_bundle(example_bundle)
    d = result.metrics.quality.details
    assert d["quality_evaluation"] == "empirical_mask_iou"
    assert result.metrics.quality.metric_name == "target_recall"
    assert result.metrics.quality.value == pytest.approx(2 / 3)
    assert d["target_recall"] == pytest.approx(2 / 3)
    assert d["detection_precision"] == pytest.approx(0.6)
    assert (d["matched_detections"], d["total_predictions"], d["false_positive_detections"]) == (
        3,
        5,
        2,
    )
    assert d["target_f1"] is None


def test_the_generated_bundle_runs_through_the_cli(tmp_path: Path) -> None:
    from aerointentbench.run_benchmark import main

    build_empirical_bundle(EXAMPLE_MANIFEST, tmp_path / "bundle", created_at="FIXED")
    output = tmp_path / "result.json"
    exit_code = main(
        [
            "--data-root",
            str(tmp_path / "bundle"),
            "--episode",
            str(tmp_path / "bundle" / "episodes" / "episode.json"),
            "--contract",
            str(tmp_path / "bundle" / "contracts" / "contract.json"),
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
    payload = json.loads(output.read_text())
    assert payload["executor_id"] == "replay"
    assert payload["episodes"][0]["quality"]["details"]["target_recall"] == pytest.approx(2 / 3)


def test_policy_visible_state_has_no_ground_truth(example_bundle: Path) -> None:
    result = _run_bundle(example_bundle)
    for step in result.record.steps:
        blob = json.dumps(step.state)
        for token in ("mask", "track_id", "ground_truth", "recall", "GT_"):
            assert token not in blob, f"{token!r} leaked into policy-visible state"


# --- validator negative checks ---------------------------------------------------------------


def test_validation_flags_a_corrupted_provenance_count(example_bundle: Path) -> None:
    provenance_path = example_bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["frame_count"] = 999
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = validate_empirical_bundle(example_bundle)
    assert not report.ok
    assert any("frame_count" in error for error in report.errors)


def test_the_predictions_parser_rejects_forbidden_fields(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "prediction_id": "P",
                "category": "person",
                "confidence": 0.5,
                "mask": MASK2,
                "track_id": "GT_X",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationError, match="track_id"):
        parse_predictions_source(path, config_id="C", width=2, height=2)


def test_the_measurements_parser_requires_a_failure_reason_for_failures(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text(
        "frame_id,success,latency_s,compute_energy_j,upload_mb,download_mb,failure_reason\n"
        "0,false,0.1,1.0,0.0,0.0,\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationError, match="must state a failure_reason"):
        parse_measurements_source(path, config_id="C")
