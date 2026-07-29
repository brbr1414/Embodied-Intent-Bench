"""V2: the visual closed loop, from scenario schema to mission metrics.

Everything runs on a tiny generated PNG world -- the large aerial rasters under
``src/v2_img`` are never required by the unit suite. The whole module skips cleanly in
an environment without the ``aerointentbench[v2]`` extras, keeping the V1 core's
zero-dependency guarantee intact.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError, SchemaVersionError  # noqa: E402
from aerointentbench.v2.camera import CameraRenderer  # noqa: E402
from aerointentbench.v2.executors import build_executors  # noqa: E402
from aerointentbench.v2.objects import ObjectLayer  # noqa: E402
from aerointentbench.v2.runner import MissionRunner, run_mission  # noqa: E402
from aerointentbench.v2.scenario import load_scenario  # noqa: E402
from aerointentbench.v2.trajectory import (  # noqa: E402
    PolylineTrajectory,
    lawnmower_waypoints,
)
from aerointentbench.v2.world import ArrayWorld, ImageWorld, resize_mask_nearest  # noqa: E402

# --- fixtures --------------------------------------------------------------------------------

BG = (120, 110, 100)  # a neutral aerial-ish gray-brown that triggers neither executor


def write_world_png(path: Path, *, width_px: int = 400, height_px: int = 300) -> Path:
    """A small world image: uniform background with a black (invalid) right strip."""
    from PIL import Image

    rgb = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    rgb[:] = BG
    rgb[:, -40:] = 0  # invalid: all-zero nodata strip
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def scenario_payload(world_path: Path, **overrides) -> dict:
    payload = {
        "scenario_schema_version": "2.0",
        "scenario_id": "V2_TEST",
        "random_seed": 7,
        "world": {
            "image_path": str(world_path),
            "source_id": "test_world",
            "meters_per_pixel": 0.5,
            "invalid_pixel_rule": "all_zero",
        },
        "drone": {
            "initial_position_m": [30.0, 40.0],
            "altitude_m": 40.0,
            "speed_mps": 5.0,
            "battery_capacity_wh": 2.0,
            "flight_power_w": 100.0,
        },
        "trajectory": {"type": "polyline", "waypoints_m": [[30.0, 40.0], [150.0, 40.0]]},
        "camera": {
            "projection": "orthographic",
            "footprint_width_m": 40.0,
            "footprint_height_m": 30.0,
            "output_width_px": 64,
            "output_height_px": 48,
        },
        "simulation": {
            "observation_interval_s": 1.0,
            "fallback_config_id": "FAST",
            "matching_iou_threshold": 0.3,
            "network": {"bandwidth_mbps": 20.0, "rtt_ms": 30.0, "packet_loss_frac": 0.0},
        },
        "objects": [
            {
                "object_id": "TGT_1",
                "class_id": "rescue_target",
                "is_target": True,
                "position_m": [50.0, 40.0],
                "width_m": 6.0,
                "height_m": 3.0,
                "rotation_deg": 0.0,
                "z_order": 10,
                "appearance": {
                    "shape": "rect",
                    "body_rgb": [230, 70, 20],
                    "stripe": True,
                    "stripe_rgb": [255, 140, 40],
                    "alpha": 1.0,
                },
            },
            {
                "object_id": "DIS_1",
                "class_id": "debris",
                "is_target": False,
                "position_m": [90.0, 40.0],
                "width_m": 6.0,
                "height_m": 3.0,
                "rotation_deg": 0.0,
                "z_order": 5,
                "appearance": {
                    "shape": "rect",
                    "body_rgb": [146, 95, 100],
                    "stripe": False,
                    "alpha": 1.0,
                },
            },
        ],
        "executor_configs": [
            {
                "config_id": "FAST",
                "model_strategy_id": "T_FAST",
                "kind": "fast_weak",
                "mission_latency_s": 0.4,
                "energy_j_per_call": 2.0,
                "communication_mb_per_call": 0.0,
                "quality_tier": "low",
                "parameters": {"downsample": 2},
            },
            {
                "config_id": "STRONG",
                "model_strategy_id": "T_STRONG",
                "kind": "slow_strong",
                "mission_latency_s": 2.5,
                "energy_j_per_call": 8.0,
                "communication_mb_per_call": 0.0,
                "quality_tier": "high",
                "parameters": {"min_component_px": 6},
            },
        ],
        "mission_contract": {
            "contract_id": "V2_TEST_CONTRACT",
            "quality_metric": "target_recall",
            "quality_operator": ">=",
            "quality_threshold": 0.5,
            "deadline_s": 60.0,
            "communication_budget_mb": 10.0,
            "min_final_battery_frac": 0.0,
        },
    }
    for dotted, value in overrides.items():
        node = payload
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return payload


@pytest.fixture
def world_png(tmp_path: Path) -> Path:
    return write_world_png(tmp_path / "world.png")


@pytest.fixture
def make_scenario(tmp_path: Path, world_png: Path):
    def _make(**overrides) -> Path:
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world_png, **overrides)), encoding="utf-8")
        return path

    return _make


def gray_world(width_px: int = 400, height_px: int = 300) -> ArrayWorld:
    rgb = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    rgb[:] = BG
    return ArrayWorld(rgb, meters_per_pixel=0.5)


# --- scenario schema -------------------------------------------------------------------------


def test_a_valid_scenario_loads(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    assert scenario.scenario_id == "V2_TEST"
    assert scenario.config_ids == ("FAST", "STRONG")
    assert len(scenario.targets) == 1 and len(scenario.distractors) == 1


def test_an_unsupported_scenario_version_fails(make_scenario) -> None:
    with pytest.raises(SchemaVersionError):
        load_scenario(make_scenario(**{"scenario_schema_version": "9.9"}))


def test_an_unknown_field_is_rejected(make_scenario) -> None:
    with pytest.raises(SchemaValidationError, match="unknown field"):
        load_scenario(make_scenario(**{"surprise": 1}))


def test_a_fallback_outside_the_configs_fails(make_scenario) -> None:
    with pytest.raises(SchemaValidationError, match="fallback_config_id"):
        load_scenario(make_scenario(**{"simulation.fallback_config_id": "NOPE"}))


def test_a_lawnmower_without_region_fails(make_scenario) -> None:
    with pytest.raises(SchemaValidationError, match="lawnmower"):
        load_scenario(make_scenario(**{"trajectory": {"type": "lawnmower"}}))


def test_an_object_outside_the_valid_region_fails(make_scenario) -> None:
    with pytest.raises(SchemaValidationError, match="outside"):
        load_scenario(make_scenario(**{"world.valid_region_m": [0.0, 0.0, 40.0, 40.0]}))


# --- world: conversions, windows, invalid regions --------------------------------------------


def test_meter_pixel_conversions_roundtrip() -> None:
    world = gray_world()
    assert world.meters_to_pixels(10.0, 5.0) == (20.0, 10.0)
    assert world.pixels_to_meters(20.0, 10.0) == (10.0, 5.0)
    assert world.width_m == 200.0 and world.height_m == 150.0


def test_window_read_matches_the_source_pixels() -> None:
    rgb = np.random.default_rng(0).integers(1, 255, (100, 100, 3), dtype=np.uint8)
    world = ArrayWorld(rgb, meters_per_pixel=1.0)
    # 20x20 m window centred at (50, 50) with 1 m/px -> exactly pixels [40:60, 40:60].
    read = world.read_window_m((50.0, 50.0), 20.0, 20.0, 20, 20)
    assert read.rgb.shape == (20, 20, 3)
    assert np.array_equal(read.rgb, rgb[40:60, 40:60])
    assert read.valid.all()


def test_out_of_bounds_window_is_padded_invalid_and_black() -> None:
    world = gray_world(100, 100)
    read = world.read_window_m((0.0, 0.0), 20.0, 20.0, 40, 40)  # top-left corner overhang
    assert not read.valid[:20, :20].any()  # outside the raster
    assert (read.rgb[~read.valid] == 0).all()
    assert read.valid[25:, 25:].all()


def test_all_zero_pixels_are_invalid_scenery(world_png: Path) -> None:
    world = ImageWorld(world_png, meters_per_pixel=0.5)
    # The black strip occupies the right 40 px = rightmost 20 m of the 200 m world.
    read = world.read_window_m((190.0, 75.0), 20.0, 20.0, 40, 40)
    assert not read.valid.all()
    assert (read.rgb[~read.valid] == 0).all()
    assert world.valid_fraction_m((100.0, 75.0), 20.0, 20.0) == 1.0


def test_nearest_neighbour_mask_resize_preserves_labels() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    resized = resize_mask_nearest(mask, 4, 4)
    assert resized.shape == (4, 4)
    assert set(np.unique(resized)) == {0, 1, 2, 3}  # nearest never invents labels
    assert resized[0, 0] == 0 and resized[3, 3] == 3


# --- trajectory ------------------------------------------------------------------------------


def test_polyline_moves_at_fixed_speed_and_clamps() -> None:
    trajectory = PolylineTrajectory(waypoints_m=((0.0, 0.0), (100.0, 0.0)), speed_mps=5.0)
    assert trajectory.position_at(0.0) == (0.0, 0.0)
    assert trajectory.position_at(4.0) == (20.0, 0.0)
    assert trajectory.duration_s == 20.0
    assert trajectory.position_at(999.0) == (100.0, 0.0)  # clamped at the end
    assert trajectory.progress_at(10.0) == 0.5
    assert trajectory.is_complete_at(20.0) and not trajectory.is_complete_at(19.9)


def test_lawnmower_generates_a_serpentine_that_covers_the_region() -> None:
    waypoints = lawnmower_waypoints((0.0, 0.0, 100.0, 30.0), lane_spacing_m=10.0)
    ys = sorted({p[1] for p in waypoints})
    assert ys == [0.0, 10.0, 20.0, 30.0]  # last lane sits on the far edge
    # Alternating direction: lane 0 ends at x=100, lane 1 starts at x=100.
    assert waypoints[1] == (100.0, 0.0) and waypoints[2] == (100.0, 10.0)


# --- rendering and ground truth --------------------------------------------------------------


def build_renderer(scenario) -> tuple[CameraRenderer, ObjectLayer]:
    world = ImageWorld(Path(scenario.world.image_path), scenario.world.meters_per_pixel)
    layer = ObjectLayer(scenario.objects)
    return (
        CameraRenderer(
            scenario_id=scenario.scenario_id, world=world, objects=layer, camera=scenario.camera
        ),
        layer,
    )


def test_the_crop_changes_with_uav_position(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    renderer, _ = build_renderer(scenario)
    at_target = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    far_away = renderer.render(position_m=(130.0, 40.0), capture_time_s=1.0, observation_id=1)
    assert not np.array_equal(at_target.rgb, far_away.rgb)
    assert at_target.footprint_m != far_away.footprint_m


def test_objects_are_composited_with_aligned_ground_truth(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    renderer, layer = build_renderer(scenario)
    obs = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    target_px = obs.semantic_gt == 1
    assert target_px.any(), "target must be visible in the crop"
    # RGB actually changed where the object is (composited, not just labelled).
    assert (obs.rgb[target_px][:, 0].astype(int) > 180).all()
    # Instance mask aligns exactly with the semantic mask for this lone target.
    assert np.array_equal(target_px, obs.instance_gt == layer.instance_id("TGT_1"))
    assert obs.visible_target_ids == ("TGT_1",)
    assert obs.visible_distractor_ids == ()


def test_boundary_clipping_keeps_partial_objects(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    renderer, _ = build_renderer(scenario)
    # Footprint x in [48, 88]: the target (x in [47, 53]) pokes in from the left edge.
    obs = renderer.render(position_m=(68.0, 40.0), capture_time_s=0.0, observation_id=0)
    target_px = obs.semantic_gt == 1
    assert target_px.any()
    assert "TGT_1" in obs.visible_target_ids
    cols = np.where(target_px.any(axis=0))[0]
    assert cols.min() == 0, "clipped object must touch the crop edge"


def test_rotation_changes_the_rendered_mask(make_scenario) -> None:
    a = load_scenario(make_scenario())
    payload_objects = scenario_payload(Path(a.world.image_path))["objects"]
    b = load_scenario(make_scenario(**{"objects": payload_objects}))
    renderer_a, _ = build_renderer(a)
    obs_a = renderer_a.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    rotated = json.loads(json.dumps(scenario_payload(Path(a.world.image_path))))
    rotated["objects"][0]["rotation_deg"] = 45.0
    path = Path(a.world.image_path).parent / "rotated.json"
    path.write_text(json.dumps(rotated), encoding="utf-8")
    renderer_b, _ = build_renderer(load_scenario(path))
    obs_b = renderer_b.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    assert not np.array_equal(obs_a.semantic_gt, obs_b.semantic_gt)
    del b


def test_overlapping_objects_resolve_by_z_order_deterministically(tmp_path: Path) -> None:
    world_path = write_world_png(tmp_path / "w.png")
    payload = scenario_payload(world_path)
    payload["objects"][1]["position_m"] = [51.0, 40.0]  # overlap the target
    payload["objects"][1]["z_order"] = 20  # distractor paints on top
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scenario = load_scenario(path)
    renderer, layer = build_renderer(scenario)
    obs = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    overlap_zone = (obs.instance_gt == layer.instance_id("DIS_1")) & (obs.semantic_gt == 2)
    assert overlap_zone.any(), "the higher z-order object owns the overlap"


def test_rendering_is_deterministic(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    renderer, _ = build_renderer(scenario)
    one = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    two = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    assert np.array_equal(one.rgb, two.rgb)
    assert np.array_equal(one.instance_gt, two.instance_gt)


# --- executors -------------------------------------------------------------------------------


def executors_for(make_scenario):
    return build_executors(load_scenario(make_scenario()).executor_configs)


def test_predictions_depend_on_rgb_content(make_scenario) -> None:
    executors = executors_for(make_scenario)
    blank = np.full((48, 64, 3), BG, dtype=np.uint8)
    marked = blank.copy()
    marked[20:30, 20:40] = (230, 70, 20)
    for executor in executors.values():
        assert not executor.run(blank).prediction_mask.any(), "blank scenery must predict nothing"
        assert executor.run(marked).prediction_mask.any(), "the marker colour must be found"


def test_fast_and_strong_disagree_on_the_distractor(make_scenario) -> None:
    executors = executors_for(make_scenario)
    distractor = np.full((48, 64, 3), BG, dtype=np.uint8)
    distractor[20:30, 20:40] = (146, 95, 100)
    assert executors["FAST"].run(distractor).prediction_mask.any(), "weak gate takes the bait"
    assert not executors["STRONG"].run(distractor).prediction_mask.any(), "strong gate does not"


def test_strong_removes_small_components(make_scenario) -> None:
    executors = executors_for(make_scenario)
    speckle = np.full((48, 64, 3), BG, dtype=np.uint8)
    speckle[10, 10] = (230, 70, 20)  # a single marker pixel: below min_component_px
    assert not executors["STRONG"].run(speckle).prediction_mask.any()


def test_executors_see_only_rgb(make_scenario) -> None:
    executors = executors_for(make_scenario)
    for executor in executors.values():
        params = list(inspect.signature(executor.run).parameters)
        assert params == ["rgb"], "an executor receives the RGB crop and nothing else"
    import aerointentbench.v2.executors as module

    source = inspect.getsource(module)
    for forbidden in ("semantic_gt", "instance_gt", "visible_target_ids", "ObjectLayer"):
        assert forbidden not in source, f"executors must not reference {forbidden}"


def test_measurement_labels_separate_simulated_from_measured(make_scenario) -> None:
    executors = executors_for(make_scenario)
    result = executors["FAST"].run(np.full((48, 64, 3), BG, dtype=np.uint8))
    assert result.mission_latency_s == 0.4  # configured, drives the mission clock
    assert result.measured_wall_clock_s > 0.0  # actually measured, diagnostic only
    assert "simulated" in result.measurement_provenance["mission_latency_s"]
    assert "measured" in result.measurement_provenance["measured_wall_clock_s"]
    assert "simulated" in result.measurement_provenance["energy_j"]


# --- the closed loop -------------------------------------------------------------------------


def test_latency_skips_scheduled_observations(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_strong")
    first = result.observations[0]
    assert first.capture_time_s == 0.0
    assert first.completion_time_s == 2.5
    assert first.skipped_after == (1, 2)  # the 1.0 s and 2.0 s captures passed while busy
    assert result.observations[1].capture_time_s == 3.0
    assert result.skipped_observation_count > 0


def test_fast_processes_every_observation(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_fast")
    assert result.skipped_observation_count == 0
    assert result.processed_observation_count >= 20


def test_the_uav_keeps_moving_during_inference(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_strong")
    first = result.observations[0]
    dx = first.completion_position_m[0] - first.capture_position_m[0]
    assert dx == pytest.approx(5.0 * 2.5)  # speed x latency: the UAV genuinely moved


def test_predictions_score_against_capture_time_ground_truth(make_scenario, tmp_path) -> None:
    # Strong latency 4.0 s: capture at t=4 sees the target; by completion (t=8) the UAV
    # is 20 m further and the target is fully outside the footprint. The match must
    # still be credited -- proof the evaluator used the capture-time observation.
    path = make_scenario(
        **{
            "executor_configs": [
                dict(scenario_payload(tmp_path / "world.png")["executor_configs"][0]),
                {
                    **scenario_payload(tmp_path / "world.png")["executor_configs"][1],
                    "mission_latency_s": 4.0,
                },
            ]
        }
    )
    scenario = load_scenario(path)
    runner = MissionRunner(scenario, "always_strong")
    result = runner.run()
    matched = [log for log in result.observations if log.score["matched_target_ids"]]
    assert matched, "the flight line passes over the target; something must match"
    scored = matched[-1]  # the last sighting: gone from view by its completion time
    completion_view = runner.render_at(scored.completion_time_s)
    assert "TGT_1" not in completion_view.visible_target_ids, (
        "the target must be gone by completion time for this test to bite"
    )
    assert scored.score["matched_target_ids"] == ["TGT_1"]


def test_battery_and_energy_accounting(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_fast")
    calls = result.processed_observation_count
    expected_energy = 100.0 * result.final_time_s + 2.0 * calls  # flight + per-call
    assert result.cumulative_energy_j == pytest.approx(expected_energy)
    capacity_j = 2.0 * 3600.0
    assert result.final_battery_frac == pytest.approx(1.0 - expected_energy / capacity_j)
    fracs = []
    # battery reported to the policy decreases monotonically across the mission
    for log in result.observations:
        fracs.append(log.execution["energy_j"])
    assert result.final_battery_frac < 1.0


def test_deadline_termination(make_scenario) -> None:
    result = run_mission(
        load_scenario(make_scenario(**{"mission_contract.deadline_s": 5.0})), "always_fast"
    )
    assert result.termination_reason == "deadline_exceeded"
    assert not result.constraints["deadline_success"] or result.final_time_s <= 5.0


def test_path_end_termination(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_fast")
    assert result.termination_reason == "path_complete"
    assert result.path_progress == 1.0


def test_battery_termination(make_scenario) -> None:
    result = run_mission(
        load_scenario(make_scenario(**{"drone.battery_capacity_wh": 0.2})), "always_fast"
    )
    assert result.termination_reason == "battery_depleted"


def test_policies_dispatch_their_executor(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    fast = run_mission(scenario, "always_fast")
    strong = run_mission(scenario, "always_strong")
    assert set(fast.config_selection_history) == {"FAST"}
    assert set(strong.config_selection_history) == {"STRONG"}


def test_an_invalid_policy_action_uses_the_scenario_fallback(make_scenario) -> None:
    from aerointentbench.policies.static import StaticPolicy

    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_fast")
    runner._policy = StaticPolicy("NOT_A_CONFIG")
    result = runner.run()
    assert set(result.config_selection_history) == {"FAST"}  # the declared fallback
    assert all(not log.action_valid for log in result.observations)


def test_missions_are_deterministic(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    one = run_mission(scenario, "always_strong").to_dict()
    two = run_mission(scenario, "always_strong").to_dict()
    for payload in (one, two):  # wall-clock diagnostics legitimately differ between runs
        for log in payload["observations"]:
            log["execution"].pop("measured_wall_clock_s")
    assert one == two


def test_mission_metrics_carry_the_v1_empirical_vocabulary(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_strong")
    q = result.quality
    for key in (
        "target_recall",
        "detection_precision",
        "false_positive_detections",
        "false_positives_per_processed_minute",
        "false_positives_per_mission_minute",
        "targets_encountered",
        "targets_detected",
        "targets_missed_while_visible",
    ):
        assert key in q, key
    assert q["target_f1"] is None
    assert q["quality_evaluation"] == "empirical_mask_iou"
    assert q["target_recall"] == 1.0  # the lone target sits on the flight line
    assert result.mission_success is True


def test_fast_pays_in_precision_strong_pays_in_coverage(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    fast = run_mission(scenario, "always_fast")
    strong = run_mission(scenario, "always_strong")
    assert fast.quality["false_positive_detections"] > 0  # the distractor fools the weak gate
    assert strong.quality["false_positive_detections"] == 0
    assert strong.skipped_observation_count > fast.skipped_observation_count
    assert fast.processed_observation_count > strong.processed_observation_count


def test_rule_based_policy_runs_on_the_v1_interface(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "rule_based")
    assert result.processed_observation_count > 0
    assert set(result.config_selection_history) <= {"FAST", "STRONG"}


# --- skip semantics: encountered-but-missed --------------------------------------------------
#
# The semantic question this section pins: does a target that was visible ONLY during
# skipped scheduled observations count as encountered-but-missed, or does it silently
# vanish from the evaluation? Desired (and implemented): it counts as encountered, is
# not detected, is reported missed-while-visible, and the slow executor is penalized --
# while the executor itself never runs on a skipped capture.
#
# Geometry: speed 6 m/s from x=30, footprint 8 m wide (half 4), target spanning
# x in [38, 40]. Footprint x-ranges per scheduled capture:
#   t=0.0 -> [26, 34]  target OUTSIDE
#   t=1.0 -> [32, 40]  target visible
#   t=2.0 -> [38, 46]  target visible
#   t=3.0 -> [44, 52]  target OUTSIDE


def skip_semantics_scenario(tmp_path: Path, world_path: Path) -> Path:
    payload = scenario_payload(world_path)
    payload["scenario_id"] = "V2_SKIP_SEMANTICS"
    payload["drone"]["speed_mps"] = 6.0
    payload["trajectory"] = {"type": "polyline", "waypoints_m": [[30.0, 40.0], [78.0, 40.0]]}
    payload["camera"] = {
        "projection": "orthographic",
        "footprint_width_m": 8.0,
        "footprint_height_m": 6.0,
        "output_width_px": 32,
        "output_height_px": 24,
    }
    payload["objects"] = [
        {
            "object_id": "TGT_SKIP",
            "class_id": "rescue_target",
            "is_target": True,
            "position_m": [39.0, 40.0],
            "width_m": 2.0,
            "height_m": 2.0,
            "rotation_deg": 0.0,
            "z_order": 1,
            "appearance": {
                "shape": "rect",
                "body_rgb": [230, 70, 20],
                "stripe": False,
                "alpha": 1.0,
            },
        }
    ]
    payload["executor_configs"][1]["mission_latency_s"] = 2.4  # STRONG
    payload["mission_contract"]["deadline_s"] = 20.0
    path = tmp_path / "skip_semantics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fast_executor_can_catch_the_short_visibility_window(tmp_path: Path, world_png) -> None:
    scenario = load_scenario(skip_semantics_scenario(tmp_path, world_png))
    result = run_mission(scenario, "always_fast")

    processed = [log.capture_time_s for log in result.observations]
    assert processed[:4] == [0.0, 1.0, 2.0, 3.0]  # latency 0.4 < interval: nothing skipped
    visible_at = {
        log.capture_time_s: log.score["visible_target_ids"] for log in result.observations
    }
    assert visible_at[0.0] == [] and visible_at[3.0] == []
    assert visible_at[1.0] == ["TGT_SKIP"] and visible_at[2.0] == ["TGT_SKIP"]

    q = result.quality
    assert q["targets_encountered"] == 1
    assert q["targets_detected"] == 1
    assert q["targets_missed_while_visible"] == 0
    assert q["target_recall"] == 1.0
    assert result.mission_success is True


def test_skipped_only_target_counts_as_encountered_but_missed(tmp_path: Path, world_png) -> None:
    scenario = load_scenario(skip_semantics_scenario(tmp_path, world_png))
    result = run_mission(scenario, "always_strong")

    # Capture at 0.0 s, latency 2.4 s: the 1.0 s and 2.0 s captures are skipped, the
    # next processed observation is 3.0 s -- and the target is out of view by then.
    first = result.observations[0]
    assert first.capture_time_s == 0.0
    assert first.completion_time_s == pytest.approx(2.4)
    assert first.skipped_after == (1, 2)
    assert first.capture_position_m == (30.0, 40.0)
    assert first.completion_position_m == pytest.approx((44.4, 40.0))  # 6 m/s x 2.4 s

    processed = [log.capture_time_s for log in result.observations]
    assert processed[:2] == [0.0, 3.0], "stale skipped captures must never be processed later"
    assert 1.0 not in processed and 2.0 not in processed
    for log in result.observations:
        assert log.score["visible_target_ids"] == [], "no processed capture ever saw the target"

    # The executor ran exactly once per processed observation -- never on a skip.
    assert len(result.observations) == result.processed_observation_count
    assert result.skipped_observation_count == len(result.skipped_observation_ids) > 0

    # The semantic core: visible only during skipped captures => encountered-but-missed,
    # not silently vanished. Recall's denominator is the scenario total either way.
    q = result.quality
    assert q["targets_encountered"] == 1, "a skip-only target must still count as encountered"
    assert q["targets_detected"] == 0
    assert q["targets_missed_while_visible"] == 1
    assert q["target_recall"] == 0.0
    assert result.mission_success is False, "the slow executor is penalized for the miss"


def test_prediction_from_t0_scores_against_t0_ground_truth(tmp_path: Path, world_png) -> None:
    scenario = load_scenario(skip_semantics_scenario(tmp_path, world_png))
    result = run_mission(scenario, "always_strong")
    first = result.observations[0]
    # The score reflects capture-time GT (t=0.0: nothing visible, nothing matched), even
    # though the target genuinely entered the camera footprint during the skipped 1.0 s
    # and 2.0 s captures -- proof the evaluation used the observation captured at 0.0
    # and never re-rendered mid-flight imagery for it.
    assert first.score["visible_target_ids"] == []
    assert first.score["matched_target_ids"] == []
    runner = MissionRunner(scenario, "always_strong")
    for skipped_time in (1.0, 2.0):
        assert "TGT_SKIP" in runner.render_at(skipped_time).visible_target_ids, (
            "the target was really there during the skipped captures"
        )


# --- CLI and visualisation -------------------------------------------------------------------


def test_cli_validate_overview_and_run(make_scenario, tmp_path: Path, capsys) -> None:
    from aerointentbench.v2.cli import main

    scenario_path = make_scenario()
    assert main(["validate", "--scenario", str(scenario_path)]) == 0
    banner = capsys.readouterr().out
    assert "V2 visual" in banner and "SIMULATED" in banner and "SYNTHETIC" in banner

    out = tmp_path / "cli_out"
    assert main(["overview", "--scenario", str(scenario_path), "--output", str(out)]) == 0
    assert (out / "V2_TEST_overview.png").is_file()

    assert (
        main(
            [
                "run",
                "--scenario",
                str(scenario_path),
                "--policy",
                "always_fast",
                "--output",
                str(out),
                "--debug-observations",
                "2",
            ]
        )
        == 0
    )
    result_path = out / "V2_TEST_always_fast.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text())
    assert payload["benchmark_mode"] == "v2_visual"
    panels = list(out.glob("V2_TEST_obs*.png"))
    assert 1 <= len(panels) <= 2


def test_cli_rejects_a_broken_scenario(make_scenario, tmp_path: Path) -> None:
    from aerointentbench.v2.cli import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scenario_schema_version": "2.0"}), encoding="utf-8")
    assert main(["validate", "--scenario", str(bad)]) == 2
