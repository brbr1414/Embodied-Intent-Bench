"""V2 replay: runtime snapshots, constraint status, bundle export, and the HTML viewer.

Everything runs on the tiny generated PNG world from ``test_v2_visual_loop`` — no local
rasters, no human assets, no real models. Skips cleanly without the ``[v2]`` extras.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.replay_export import (
    REPLAY_SCHEMA_VERSION,
    STATUS_AT_RISK,
    STATUS_NOT_APPLICABLE,
    STATUS_SAFE,
    STATUS_UNKNOWN,
    STATUS_VIOLATED,
    battery_status,
    communication_status,
    deadline_status,
    export_replay_bundle,
    overall_status,
    quality_status,
)
from aerointentbench.v2.runner import MissionRunner, run_mission
from aerointentbench.v2.scenario import load_scenario
from tests.test_v2_visual_loop import scenario_payload, write_world_png

#: The exact policy-visible RuntimeState field set (mirrors tests/test_runtime_state.py).
POLICY_VISIBLE_FIELDS = {
    "current_time_s",
    "frame_id",
    "battery_frac",
    "power_mode",
    "network",
    "current_config_id",
    "remaining_deadline_s",
    "cumulative_energy_j",
    "cumulative_communication_mb",
    "path_progress",
    "evidence_summary",
}


@pytest.fixture
def make_scenario(tmp_path: Path):
    world = write_world_png(tmp_path / "world.png")

    def _make(**overrides) -> Path:
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world, **overrides)), encoding="utf-8")
        return path

    return _make


@pytest.fixture
def bundle(make_scenario, tmp_path: Path) -> Path:
    return export_replay_bundle(make_scenario(), "always_strong", tmp_path / "bundle")


# --- runtime snapshots -----------------------------------------------------------------------


def test_snapshots_are_off_by_default(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_fast")
    assert all(log.runtime is None for log in result.observations)
    assert all("runtime" not in log.to_dict() for log in result.observations)


def test_every_processed_observation_gets_one_snapshot(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_strong", record_runtime_snapshots=True)
    result = runner.run()
    assert len(result.observations) == result.processed_observation_count
    assert all(log.runtime is not None for log in result.observations)

    captures = [log.capture_time_s for log in result.observations]
    assert captures == sorted(captures) and len(set(captures)) == len(captures)
    for log in result.observations:
        assert log.capture_position_m == runner.trajectory.position_at(log.capture_time_s)
        assert log.runtime["at_capture"]["current_time_s"] == log.capture_time_s


def test_snapshot_preserves_configs_and_records_switches(make_scenario) -> None:
    class Alternator:
        def __init__(self) -> None:
            self.calls = 0

        def select_config(self, contract, runtime_state, allowed_configs) -> str:
            self.calls += 1
            return "FAST" if self.calls % 2 else "STRONG"

    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_fast", record_runtime_snapshots=True)
    runner._policy = Alternator()
    result = runner.run()

    logs = result.observations
    assert len(logs) >= 3
    assert all(log.requested_config_id == log.executed_config_id for log in logs)
    assert logs[0].runtime["config_switched"] is False  # nothing to switch from
    for prev, cur in itertools.pairwise(logs):
        expected = cur.executed_config_id != prev.executed_config_id
        assert cur.runtime["config_switched"] is expected


def test_snapshot_records_fallback_and_invalid_action(make_scenario) -> None:
    from aerointentbench.policies.static import StaticPolicy

    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_fast", record_runtime_snapshots=True)
    runner._policy = StaticPolicy("NOT_A_CONFIG")
    result = runner.run()
    assert result.observations
    for log in result.observations:
        assert log.action_valid is False
        assert log.runtime["fallback_used"] is True
        assert log.executed_config_id == "FAST"  # the scenario's declared fallback


def test_snapshot_records_frame_skips(make_scenario) -> None:
    result = run_mission(
        load_scenario(make_scenario()), "always_strong", record_runtime_snapshots=True
    )
    skipped_from_logs = [i for log in result.observations for i in log.skipped_after]
    assert skipped_from_logs, "the slow executor must skip captures"
    assert tuple(skipped_from_logs) == result.skipped_observation_ids


def test_snapshot_resources_move_monotonically(make_scenario) -> None:
    result = run_mission(
        load_scenario(make_scenario()), "always_strong", record_runtime_snapshots=True
    )
    afters = [log.runtime["after_completion"] for log in result.observations]
    for prev, cur in itertools.pairwise(afters):
        assert cur["cumulative_energy_j"] >= prev["cumulative_energy_j"]
        assert cur["cumulative_communication_mb"] >= prev["cumulative_communication_mb"]
        assert cur["battery_frac"] <= prev["battery_frac"]
        assert cur["remaining_deadline_s"] <= prev["remaining_deadline_s"]


def test_snapshot_at_capture_is_exactly_the_policy_visible_state(make_scenario) -> None:
    result = run_mission(
        load_scenario(make_scenario()), "always_fast", record_runtime_snapshots=True
    )
    for log in result.observations:
        at_capture = log.runtime["at_capture"]
        assert set(at_capture) == POLICY_VISIBLE_FIELDS
        # Evidence stays prediction-derived; no ground-truth key may appear.
        assert set(at_capture["evidence_summary"]) == {
            "predicted_unique_targets",
            "processed_frames",
            "mean_prediction_confidence",
        }


# --- constraint status -----------------------------------------------------------------------


def test_deadline_status_levels() -> None:
    assert deadline_status(-0.1, 60.0) == STATUS_VIOLATED
    assert deadline_status(5.0, 60.0) == STATUS_AT_RISK  # under 20% remaining
    assert deadline_status(30.0, 60.0) == STATUS_SAFE


def test_battery_status_levels() -> None:
    assert battery_status(0.10, 0.20) == STATUS_VIOLATED
    assert battery_status(0.25, 0.20) == STATUS_AT_RISK  # within 0.1 of the floor
    assert battery_status(0.80, 0.20) == STATUS_SAFE


def test_communication_status_levels() -> None:
    assert communication_status(11.0, 10.0) == STATUS_VIOLATED
    assert communication_status(9.0, 10.0) == STATUS_AT_RISK  # >= 80% spent
    assert communication_status(1.0, 10.0) == STATUS_SAFE
    assert communication_status(0.0, 0.0) == STATUS_SAFE  # zero budget, zero use


def test_quality_status_levels(make_scenario) -> None:
    contract = load_scenario(make_scenario()).contract  # target_recall >= 0.5
    assert quality_status(contract, None) == STATUS_UNKNOWN
    assert quality_status(contract, 0.0) == STATUS_AT_RISK  # not a verdict mid-mission
    assert quality_status(contract, 1.0) == STATUS_SAFE


def test_overall_status_precedence() -> None:
    assert overall_status({"a": STATUS_SAFE, "b": STATUS_VIOLATED}) == STATUS_VIOLATED
    assert overall_status({"a": STATUS_AT_RISK, "b": STATUS_SAFE}) == STATUS_AT_RISK
    assert overall_status({"a": STATUS_UNKNOWN, "b": STATUS_SAFE}) == STATUS_UNKNOWN
    assert overall_status({"a": STATUS_SAFE, "b": STATUS_NOT_APPLICABLE}) == STATUS_SAFE


def test_final_status_never_contradicts_the_evaluator(make_scenario, tmp_path: Path) -> None:
    # An impossible end-of-mission battery floor forces a real constraint failure.
    scenario_path = make_scenario(**{"mission_contract.min_final_battery_frac": 0.99})
    bundle_dir = export_replay_bundle(scenario_path, "always_strong", tmp_path / "fail_bundle")
    result = run_mission(load_scenario(scenario_path), "always_strong")
    final = json.loads((bundle_dir / "events.json").read_text())["final"]

    assert final["mission_success"] is result.mission_success is False
    assert final["constraints"] == result.constraints
    status = final["constraint_status"]
    assert status["battery"] == STATUS_VIOLATED
    assert status["overall"] == STATUS_VIOLATED
    for name, key in [
        ("quality", "quality_success"),
        ("deadline", "deadline_success"),
        ("battery", "battery_constraint_success"),
        ("communication", "communication_constraint_success"),
    ]:
        assert (status[name] == STATUS_SAFE) is result.constraints[key]
    assert status["privacy"] == STATUS_NOT_APPLICABLE


# --- bundle export ---------------------------------------------------------------------------


def test_bundle_contains_required_files(bundle: Path) -> None:
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "events.json").is_file()
    assert (bundle / "overview.png").is_file()
    assert (bundle / "map_background.png").is_file()
    assert (bundle / "index.html").is_file()
    assert list((bundle / "frames").glob("*.png"))


def test_map_background_is_registered_to_the_mission_extent(bundle: Path) -> None:
    from PIL import Image

    manifest = json.loads((bundle / "manifest.json").read_text())
    background = manifest["map"]["background"]
    assert background["file"] == manifest["files"]["map_background"] == "map_background.png"
    x0, y0, x1, y1 = background["extent_m"]
    assert x1 > x0 and y1 > y0
    # The extent must cover the whole trajectory (it is the viewer's coordinate frame).
    for wx, wy in manifest["map"]["waypoints_m"]:
        assert x0 <= wx <= x1 and y0 <= wy <= y1
    with Image.open(bundle / background["file"]) as image:
        assert image.size == (background["width_px"], background["height_px"])


def _point_at_distance(waypoints: list[list[float]], distance: float) -> tuple[float, float]:
    """The viewer's interpolation rule, restated: constant speed along the polyline."""
    travelled = 0.0
    for (ax, ay), (bx, by) in itertools.pairwise(waypoints):
        segment = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if distance <= travelled + segment or segment == 0.0:
            fraction = 0.0 if segment == 0.0 else (distance - travelled) / segment
            fraction = min(max(fraction, 0.0), 1.0)
            return (ax + (bx - ax) * fraction, ay + (by - ay) * fraction)
        travelled += segment
    return (waypoints[-1][0], waypoints[-1][1])


def test_manifest_interpolation_basis_reproduces_the_trajectory(bundle: Path) -> None:
    """The viewer's smooth motion must land exactly on the simulator's positions."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    events = json.loads((bundle / "events.json").read_text())["events"]
    speed = manifest["map"]["drone_speed_mps"]
    waypoints = manifest["map"]["waypoints_m"]
    assert speed > 0 and manifest["map"]["trajectory_duration_s"] > 0
    for event in events:
        expected = _point_at_distance(waypoints, speed * event["capture_time_s"])
        assert expected == pytest.approx(tuple(event["capture_position_m"]), abs=1e-9)
        expected = _point_at_distance(waypoints, speed * event["completion_time_s"])
        assert expected == pytest.approx(tuple(event["completion_position_m"]), abs=1e-9)


def test_bundle_declares_the_replay_schema_version(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    events = json.loads((bundle / "events.json").read_text())
    assert manifest["replay_schema_version"] == REPLAY_SCHEMA_VERSION == "1.0"
    assert events["replay_schema_version"] == REPLAY_SCHEMA_VERSION


def test_events_match_the_mission_and_link_valid_frames(bundle: Path, make_scenario) -> None:
    events_doc = json.loads((bundle / "events.json").read_text())
    events = events_doc["events"]
    manifest = json.loads((bundle / "manifest.json").read_text())

    result = run_mission(load_scenario(make_scenario()), "always_strong")
    assert [e["observation_id"] for e in events] == [
        log.observation_id for log in result.observations
    ]
    assert [e["event_index"] for e in events] == list(range(len(events)))
    assert manifest["counts"]["events"] == len(events)

    for event in events:
        for ref in event["frames"].values():
            assert not Path(ref).is_absolute() and ".." not in ref
            assert (bundle / ref).is_file()
    assert (bundle / events[0]["frames"]["rgb"]).is_file()
    assert (bundle / events[-1]["frames"]["gt_debug"]).is_file()


def test_bundle_json_is_strict(bundle: Path) -> None:
    for name in ("events.json", "manifest.json"):
        text = (bundle / name).read_text()
        json.loads(text)  # standards-compliant
        assert not re.search(r"\bNaN\b|\bInfinity\b", text)


def _strip_wall_clock(node):
    if isinstance(node, dict):
        return {k: _strip_wall_clock(v) for k, v in node.items() if k != "measured_wall_clock_s"}
    if isinstance(node, list):
        return [_strip_wall_clock(v) for v in node]
    return node


def test_export_is_deterministic_apart_from_wall_clock(make_scenario, tmp_path: Path) -> None:
    scenario = make_scenario()
    a = export_replay_bundle(scenario, "always_strong", tmp_path / "a")
    b = export_replay_bundle(scenario, "always_strong", tmp_path / "b")
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()
    ev_a = _strip_wall_clock(json.loads((a / "events.json").read_text()))
    ev_b = _strip_wall_clock(json.loads((b / "events.json").read_text()))
    assert ev_a == ev_b
    assert (a / "map_background.png").read_bytes() == (b / "map_background.png").read_bytes()
    for frame in sorted((a / "frames").glob("*.png")):
        assert frame.read_bytes() == (b / "frames" / frame.name).read_bytes()


def test_missing_world_image_is_an_actionable_error(make_scenario, tmp_path: Path) -> None:
    scenario_path = make_scenario(**{"world.image_path": str(tmp_path / "absent.tif")})
    with pytest.raises(SchemaValidationError, match="not found"):
        export_replay_bundle(scenario_path, "always_fast", tmp_path / "nope")


def test_per_event_privacy_is_not_applicable(bundle: Path) -> None:
    events = json.loads((bundle / "events.json").read_text())["events"]
    for event in events:
        assert event["constraint_status"]["privacy"]["status"] == STATUS_NOT_APPLICABLE


# --- viewer ----------------------------------------------------------------------------------


def _embedded_payload(html: str) -> dict:
    match = re.search(
        r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert match, "the viewer must embed its replay data"
    return json.loads(match.group(1))


def test_viewer_embeds_the_replay_data(bundle: Path) -> None:
    payload = _embedded_payload((bundle / "index.html").read_text())
    events = json.loads((bundle / "events.json").read_text())
    assert payload["replay"] == events
    assert payload["manifest"]["replay_schema_version"] == REPLAY_SCHEMA_VERSION
    embedded = payload["replay"]["events"]
    assert len(embedded) == payload["manifest"]["counts"]["events"]
    assert (bundle / embedded[0]["frames"]["rgb"]).is_file()
    assert (bundle / embedded[-1]["frames"]["rgb"]).is_file()


def test_viewer_has_required_panels_and_controls(bundle: Path) -> None:
    html = (bundle / "index.html").read_text()
    for element_id in (
        "mission-status-banner",
        "mission-map",
        "perception-view",
        "mission-dashboard",
        "btn-prev",
        "btn-next",
        "btn-play",
        "timeline-slider",
        "obs-counter",
        "speed-select",
        "config-strip",
        "toggle-gt",
    ):
        assert f'id="{element_id}"' in html, f"missing #{element_id}"
    assert "Debug GT" in html  # GT is debug-only, and labelled as such
    # Hierarchy: the dashboard names perception explicitly subordinate.
    assert "subordinate to mission outcome" in html


def test_viewer_uses_the_map_background_and_continuous_playback(bundle: Path) -> None:
    html = (bundle / "index.html").read_text()
    payload = _embedded_payload(html)
    background = payload["manifest"]["map"]["background"]
    assert (bundle / background["file"]).is_file()
    assert payload["manifest"]["map"]["drone_speed_mps"] > 0
    # The playback loop is mission-time based, not per-observation stepping.
    assert "requestAnimationFrame" in html
    assert "pointAtDistance" in html
