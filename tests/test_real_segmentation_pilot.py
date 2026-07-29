"""The real-segmentation pilot *tooling* -- verified with a stub model, never a real one.

There is no real dataset, model, or GPU in this environment, so these tests exercise the
reusable conversion, measurement, and bundle-assembly tooling using a clearly-labelled
StubSegmentationModel over a tiny in-memory dataset. They prove the pipeline is correct; they
do not measure any model's quality, and produce no committed bundle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aerointentbench.benchmark import BenchmarkData, run_episode
from aerointentbench.policies.static import StaticPolicy
from aerointentbench.schemas.contract import load_contract
from aerointentbench.schemas.episode import load_episode
from aerointentbench.tasks.human_search_segmentation import load_any_ground_truth
from aerointentbench.tasks.human_search_segmentation.masks import decode_mask
from aerointentbench.tools.bundle_manifest import (
    parse_ground_truth_source,
    parse_predictions_source,
)
from aerointentbench.tools.validate_empirical_bundle import validate_empirical_bundle
from experiments.real_segmentation_pilot.build_pilot import ConfigSpec, run_pilot
from experiments.real_segmentation_pilot.convert import (
    stable_prediction_id,
    write_ground_truth_source,
)
from experiments.real_segmentation_pilot.interfaces import (
    Frame,
    GroundTruthInstance,
    InMemoryDatasetAdapter,
    PredictedInstance,
    RealSegmentationBackend,
    StubSegmentationModel,
)
from experiments.real_segmentation_pilot.masks import resize_nearest
from experiments.real_segmentation_pilot.measure import (
    EnergyEstimate,
    EnergyProvenance,
    estimate_energy,
    latency_stats,
)


def bm(*rows: str):
    return decode_mask({"height": len(rows), "width": len(rows[0]), "rows": list(rows)})


GT_A = bm(
    "00000000", "01111000", "01111000", "01111000", "01111000", "00000000", "00000000", "00000000"
)
GT_B = bm(
    "00000000", "00000000", "00000000", "00000000", "00000000", "00000111", "00000111", "00000111"
)
GT_C = GT_A
GT_IGNORE = bm(
    "00000011", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "00000000"
)
P_B_PARTIAL = bm(
    "00000000", "00000000", "00000000", "00000000", "00000000", "00000110", "00000110", "00000110"
)
P_FP = bm(
    "00000000", "00000000", "00000000", "00000000", "00000000", "00000000", "11000000", "11000000"
)
P_C_WEAK = bm(
    "00000000", "01100000", "01100000", "00000000", "00000000", "00000000", "00000000", "00000000"
)


def person(mask, confidence=0.9):
    return PredictedInstance(category="person", confidence=confidence, mask=mask)


def make_adapter() -> InMemoryDatasetAdapter:
    frames = tuple(Frame(frame_id=i) for i in (0, 1, 2))
    gt = (
        GroundTruthInstance(0, "GT_A", "person", GT_A),
        GroundTruthInstance(0, "GT_B", "person", GT_B),
        GroundTruthInstance(0, "GT_IGNORE", "person", GT_IGNORE, ignore=True),
        GroundTruthInstance(1, "GT_A", "person", GT_A),
        GroundTruthInstance(2, "GT_C", "person", GT_C),
    )
    return InMemoryDatasetAdapter("PILOT_STREAM", 8, 8, frames, gt)


STRONG = StubSegmentationModel(
    "CFG_LOCAL_STRONG",
    "stub-strong",
    {
        0: (person(GT_A), person(P_B_PARTIAL), person(P_FP)),
        1: (person(GT_A),),
        2: (person(P_C_WEAK),),
    },
)
LIGHT = StubSegmentationModel("CFG_LOCAL_LIGHT", "stub-light", {0: (person(GT_A),)})


class FakeClock:
    """A monotonic clock advancing a fixed step per call, for deterministic timing tests."""

    def __init__(self, step: float = 0.01) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def build(tmp_path: Path, *, contract=None):
    return run_pilot(
        make_adapter(),
        [
            (STRONG, ConfigSpec("CFG_LOCAL_STRONG", "stub-strong", quality_tier="high")),
            (LIGHT, ConfigSpec("CFG_LOCAL_LIGHT", "stub-light", quality_tier="low")),
        ],
        sources_dir=tmp_path / "sources",
        bundle_dir=tmp_path / "bundle",
        bundle_id="PILOT_TEST",
        average_power_w=15.0,
        energy_source="assumed 15 W board TDP (test)",
        data_origin="hand_authored_fixture",
        contract=contract
        or {"quality_metric": "target_recall", "quality_threshold": 0.5, "deadline_s": 3.0},
        created_at="FIXED",
        clock=FakeClock(),
    )


# --- adapter / GT conversion (points 1, 2) ---------------------------------------------------


def test_adapter_produces_valid_ordered_frame_ids() -> None:
    adapter = make_adapter()
    assert [f.frame_id for f in adapter.frames()] == [0, 1, 2]


def test_gt_conversion_preserves_dimensions_and_categories(tmp_path: Path) -> None:
    path = tmp_path / "gt.json"
    count = write_ground_truth_source(
        path, frame_stream_id="S", instances=make_adapter().ground_truth()
    )
    assert count == 5
    # The source parses through the builder's own ground-truth parser, dimensions preserved.
    instances = parse_ground_truth_source(path, width=8, height=8)
    assert {i.track_id for i in instances} == {"GT_A", "GT_B", "GT_C", "GT_IGNORE"}
    assert all(i.category == "person" for i in instances)
    assert all(i.mask.height == 8 and i.mask.width == 8 for i in instances)
    assert next(i for i in instances if i.track_id == "GT_IGNORE").ignore is True


def test_built_ground_truth_loads_and_excludes_ignore(tmp_path: Path) -> None:
    result = build(tmp_path)
    gt = load_any_ground_truth(result.bundle.output_root / "ground_truth" / "ground_truth.json")
    assert gt.target_track_ids() == frozenset({"GT_A", "GT_B", "GT_C"})  # GT_IGNORE excluded


# --- prediction conversion (points 3, 4, 5) --------------------------------------------------


def test_prediction_conversion_produces_valid_canonical_masks(tmp_path: Path) -> None:
    build(tmp_path)
    predictions = parse_predictions_source(
        tmp_path / "sources" / "predictions_CFG_LOCAL_STRONG.jsonl",
        config_id="CFG_LOCAL_STRONG",
        width=8,
        height=8,
    )
    assert len(predictions) == 5  # 3 + 1 + 1
    assert all(p.mask.height == 8 and p.mask.width == 8 for p in predictions)


def test_prediction_records_contain_no_ground_truth_identity(tmp_path: Path) -> None:
    build(tmp_path)
    import json

    for line in (
        (tmp_path / "sources" / "predictions_CFG_LOCAL_STRONG.jsonl").read_text().splitlines()
    ):
        record = json.loads(line)
        assert set(record) == {"frame_id", "prediction_id", "category", "confidence", "mask"}


def test_stable_prediction_ids_are_deterministic(tmp_path: Path) -> None:
    assert stable_prediction_id("C", 3, 1) == "C__f000003__i001"
    build(tmp_path / "a")
    build(tmp_path / "b")
    a = (tmp_path / "a" / "sources" / "predictions_CFG_LOCAL_STRONG.jsonl").read_text()
    b = (tmp_path / "b" / "sources" / "predictions_CFG_LOCAL_STRONG.jsonl").read_text()
    assert a == b


# --- measurement (points 6, 7) ---------------------------------------------------------------


def test_latency_records_are_finite_and_non_negative(tmp_path: Path) -> None:
    result = build(tmp_path)
    strong = next(r for r in result.per_config if r.config_id == "CFG_LOCAL_STRONG")
    assert strong.latency["count"] == 3
    assert strong.latency["min_s"] >= 0.0
    assert strong.latency["p95_s"] >= strong.latency["median_s"]


def test_latency_stats_percentiles() -> None:
    stats = latency_stats([0.01, 0.02, 0.03, 0.04, 0.10])
    assert stats.min_s == 0.01 and stats.max_s == 0.10
    assert stats.median_s == 0.03
    assert stats.p95_s == 0.10


def test_energy_provenance_is_mandatory_and_estimate_is_labelled() -> None:
    estimate = estimate_energy(0.1, average_power_w=15.0, source="datasheet")
    assert estimate.provenance is EnergyProvenance.ESTIMATED
    assert estimate.joules == pytest.approx(1.5)
    assert estimate.assumed_power_w == 15.0
    # An energy value cannot exist without a provenance source.
    with pytest.raises(ValueError, match="source"):
        EnergyEstimate(joules=1.0, provenance=EnergyProvenance.MEASURED, source="  ")


def test_energy_provenance_is_recorded_in_the_bundle(tmp_path: Path) -> None:
    build(tmp_path)
    import json

    provenance = json.loads((tmp_path / "bundle" / "provenance.json").read_text())
    assert provenance["configurations"][0]["measurement_provenance"] == "estimated"
    assert provenance["data_origin"] == "hand_authored_fixture"  # a stub run, honestly labelled


# --- mask resize -----------------------------------------------------------------------------


def test_resize_nearest_identity_and_upscale() -> None:
    small = bm("10", "01")
    assert resize_nearest(small, 2, 2) is small
    up = resize_nearest(small, 4, 4)
    assert up.height == 4 and up.width == 4
    assert up.pixels == frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)})


def test_resize_nearest_downscale() -> None:
    big = bm("1100", "1100", "0000", "0000")
    down = resize_nearest(big, 2, 2)
    assert down.height == 2 and down.width == 2
    assert down.pixels == frozenset({(0, 0)})


# --- bundle build + validation + benchmark (points 8, 9, 10) ---------------------------------


def test_two_configurations_are_distinguishable_in_metadata(tmp_path: Path) -> None:
    import json

    build(tmp_path)
    manifest = json.loads((tmp_path / "sources" / "manifest.json").read_text())
    configs = {c["config_id"]: c for c in manifest["configurations"]}
    assert configs["CFG_LOCAL_STRONG"]["model_id"] != configs["CFG_LOCAL_LIGHT"]["model_id"]
    assert configs["CFG_LOCAL_STRONG"]["quality_tier"] != configs["CFG_LOCAL_LIGHT"]["quality_tier"]


def test_generated_sources_pass_the_existing_builder_and_validator(tmp_path: Path) -> None:
    result = build(tmp_path)  # run_pilot builds with validate=True internally
    report = validate_empirical_bundle(result.bundle.output_root)
    assert report.ok, report.errors
    assert result.bundle.replay_record_count == 6  # 3 frames x 2 configs, strict


def test_the_pilot_bundle_runs_through_replay_with_expected_metrics(tmp_path: Path) -> None:
    result = build(tmp_path)
    bundle = result.bundle.output_root
    data = BenchmarkData(bundle)
    episode = load_episode(bundle / "episodes" / "episode.json")
    contract = load_contract(bundle / "contracts" / "contract.json")

    strong = run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_LIGHT",
    )
    light = run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_LIGHT"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_LIGHT",
    )
    # STRONG finds GT_A and GT_B (2/3); LIGHT finds only GT_A (1/3). Stub numbers, not a model.
    assert strong.metrics.quality.details["target_recall"] == pytest.approx(2 / 3)
    assert strong.metrics.quality.details["detection_precision"] == pytest.approx(0.6)
    assert light.metrics.quality.details["target_recall"] == pytest.approx(1 / 3)
    assert strong.metrics.quality.details["quality_evaluation"] == "empirical_mask_iou"


def test_policy_visible_state_has_no_ground_truth(tmp_path: Path) -> None:
    import json

    result = build(tmp_path)
    bundle = result.bundle.output_root
    data = BenchmarkData(bundle)
    episode = load_episode(bundle / "episodes" / "episode.json")
    contract = load_contract(bundle / "contracts" / "contract.json")
    run = run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_LIGHT",
    )
    for step in run.record.steps:
        blob = json.dumps(step.state)
        for token in ("mask", "track_id", "ground_truth", "recall", "GT_"):
            assert token not in blob


# --- the real backend stub -------------------------------------------------------------------


def test_the_real_backend_construction_raises() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        RealSegmentationBackend()
