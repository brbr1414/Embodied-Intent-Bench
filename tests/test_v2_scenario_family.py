"""V2.4: deterministic scenario-family generation and the multi-seed statistics layer.

Everything here is CI-safe: generation works from the committed base scenario JSON and
touches no rasters, no assets, and no Torch; the statistics functions are exercised on
synthetic run records. Real-model batch evaluation happens through the CLI locally.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.multi_seed_eval import (
    aggregate_records,
    fragility_report,
    paired_comparison,
    run_record,
    select_representatives,
)
from aerointentbench.v2.scenario import load_scenario
from aerointentbench.v2.scenario_family import FAMILY_VERSION, variant_id, write_variant
from aerointentbench.v2.trajectory import build_trajectory

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data" / "v2_scenarios" / "demo_img1_hard_tradeoff.json"


@pytest.fixture(scope="module")
def base_scenario():
    return load_scenario(BASE)


def make_variant(tmp_path: Path, seed: int) -> Path:
    return write_variant(BASE, seed, tmp_path / f"seed_{seed:04d}.json")


# --- deterministic generation ----------------------------------------------------------------


def test_same_seed_generates_identical_bytes(tmp_path: Path) -> None:
    a = write_variant(BASE, 7, tmp_path / "a.json")
    b = write_variant(BASE, 7, tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()


def test_different_seeds_differ_meaningfully(tmp_path: Path) -> None:
    a = json.loads(make_variant(tmp_path, 0).read_text())
    b = json.loads(make_variant(tmp_path, 1).read_text())
    positions_a = [o["position_m"] for o in a["objects"]]
    positions_b = [o["position_m"] for o in b["objects"]]
    assert positions_a != positions_b
    assert a["scenario_id"] != b["scenario_id"]


def test_variant_id_format() -> None:
    assert variant_id("V2_IMG1_HARD_TRADEOFF", 7) == "V2_IMG1_HARD_TRADEOFF_SEED_0007"


def test_negative_seed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="seed"):
        write_variant(BASE, -1, tmp_path / "bad.json")


def test_variant_json_is_strict(tmp_path: Path) -> None:
    text = make_variant(tmp_path, 3).read_text()
    json.loads(text)
    assert not re.search(r"\bNaN\b|\bInfinity\b", text)


# --- structural invariants -------------------------------------------------------------------


def test_variant_preserves_the_frozen_structure(tmp_path: Path, base_scenario) -> None:
    scenario = load_scenario(make_variant(tmp_path, 11))
    assert scenario.scenario_id == variant_id(base_scenario.scenario_id, 11)
    assert scenario.random_seed == 11
    # The trade-off levers are NOT sampled: executors, contract, trajectory, camera.
    assert scenario.config_ids == base_scenario.config_ids
    assert {s.parameters["latency_mode"] for s in scenario.executor_configs} == {"configured"}
    assert scenario.contract == base_scenario.contract
    assert scenario.trajectory == base_scenario.trajectory
    assert scenario.camera == base_scenario.camera
    assert scenario.simulation == base_scenario.simulation
    # Initial battery (always 1.0 in V2) clears the contract floor by construction.
    assert scenario.contract.min_final_battery_frac < 1.0
    assert len(scenario.targets) == 8
    assert all(o.render_mode == "image_asset" for o in scenario.targets)
    assert len(scenario.distractors) == 1


def test_sampled_values_are_recorded_and_bounded(tmp_path: Path, base_scenario) -> None:
    for seed in (0, 5, 23):
        payload = json.loads(make_variant(tmp_path, seed).read_text())
        family = payload["provenance"]["scenario_family"]
        assert family["family_version"] == FAMILY_VERSION
        assert family["seed"] == seed
        assert family["base_scenario_id"] == base_scenario.scenario_id
        assert 6.60 <= family["battery_capacity_wh"] <= 7.00
        assert payload["drone"]["battery_capacity_wh"] == family["battery_capacity_wh"]

        sampled = family["targets"]
        assert len(sampled) == 8
        smalls = [t for t in sampled if t["object_id"].startswith("TGT_S")]
        lates = [t for t in sampled if t["object_id"].startswith("TGT_E")]
        assert len(smalls) == len(lates) == 4
        for target in smalls:
            assert target["capture_slot_s"] % 2 == 0  # window covers a strong capture
            assert abs(target["time_jitter_s"]) <= 0.3
            assert 0.70 <= target["height_m"] <= 0.80  # the validated 32 px band
        for target in lates:
            assert target["capture_slot_s"] % 2 == 1  # window misses the strong cadence
            assert abs(target["time_jitter_s"]) <= 0.2
        for target in sampled:
            assert abs(target["lateral_offset_m"]) <= 1.2
            assert abs(target["rotation_deg"]) <= 45.0
            assert target["centre_time_s"] == pytest.approx(
                target["capture_slot_s"] + target["time_jitter_s"], abs=1e-6
            )
        # Distinct slots keep targets >= 2 s apart within each class.
        assert len({t["capture_slot_s"] for t in smalls}) == 4
        assert len({t["capture_slot_s"] for t in lates}) == 4


def test_target_positions_match_the_trajectory(tmp_path: Path, base_scenario) -> None:
    payload = json.loads(make_variant(tmp_path, 9).read_text())
    trajectory = build_trajectory(base_scenario.trajectory, base_scenario.drone.speed_mps)
    by_id = {o["object_id"]: o for o in payload["objects"]}
    region = base_scenario.world.valid_region_m
    for target in payload["provenance"]["scenario_family"]["targets"]:
        on_path = trajectory.position_at(target["centre_time_s"])
        stored = by_id[target["object_id"]]["position_m"]
        assert stored[0] == pytest.approx(on_path[0], abs=2e-3)
        assert stored[1] == pytest.approx(on_path[1] + target["lateral_offset_m"], abs=2e-3)
        assert region[0] <= stored[0] <= region[2] and region[1] <= stored[1] <= region[3]


# --- statistics functions on synthetic records -----------------------------------------------


def make_record(
    seed: int,
    policy: str,
    success: bool,
    *,
    quality_ok: bool | None = None,
    battery_ok: bool | None = None,
    recall: float = 0.75,
    battery: float = 0.25,
    margin: float = 0.03,
    switches: int = 1,
) -> dict:
    return {
        "scenario_id": f"S_{seed:04d}",
        "seed": seed,
        "policy": policy,
        "mission_success": success,
        "constraints": {
            "quality_success": success if quality_ok is None else quality_ok,
            "deadline_success": True,
            "battery_constraint_success": success if battery_ok is None else battery_ok,
            "communication_constraint_success": True,
        },
        "target_recall": recall,
        "detection_precision": 0.5,
        "false_positive_detections": 3,
        "final_battery_frac": battery,
        "battery_margin_frac": margin,
        "total_energy_j": 10_000.0,
        "compute_energy_j": 4_000.0,
        "skipped_observations": 10,
        "config_switch_count": switches,
        "small_targets_found": 3,
        "small_targets_total": 4,
        "late_targets_found": 3,
        "late_targets_total": 4,
    }


LIGHT, STRONG, RULE = "always_light_real", "always_strong_real", "rule_based"


def paired_records() -> list[dict]:
    rows: list[dict] = []
    # seed 0: reference pattern (light F, strong F, rule S)
    rows += [
        make_record(0, LIGHT, False),
        make_record(0, STRONG, False),
        make_record(0, RULE, True),
    ]
    # seed 1: all succeed
    rows += [make_record(1, p, True) for p in (LIGHT, STRONG, RULE)]
    # seed 2: all fail
    rows += [make_record(2, p, False) for p in (LIGHT, STRONG, RULE)]
    # seed 3: adaptive fails while light succeeds (counterexample)
    rows += [
        make_record(3, LIGHT, True),
        make_record(3, STRONG, False),
        make_record(3, RULE, False),
    ]
    # seed 4: fragile adaptive success (battery margin below threshold)
    rows += [
        make_record(4, LIGHT, False),
        make_record(4, STRONG, False),
        make_record(4, RULE, True, margin=0.004),
    ]
    return rows


def test_aggregate_counts_and_wilson_interval() -> None:
    aggregate = aggregate_records(paired_records())
    rule = aggregate[RULE]
    assert rule["evaluated_scenarios"] == 5
    assert rule["mission_success_count"] == 3
    assert rule["mission_success_rate"] == pytest.approx(0.6)
    low, high = rule["mission_success_ci_95"]
    assert 0.0 < low < 0.6 < high < 1.0
    assert rule["failure_rates"]["quality"] == pytest.approx(0.4)
    assert rule["metrics"]["target_recall"]["n"] == 5
    assert rule["metrics"]["battery_margin_frac"]["min"] == pytest.approx(0.004)


def test_paired_comparison_patterns_and_counterexamples() -> None:
    paired = paired_comparison(paired_records())
    patterns = {name: entry["seeds"] for name, entry in paired["patterns"].items()}
    assert patterns["adaptive_succeeds_both_static_fail"] == [0, 4]
    assert patterns["all_succeed"] == [1]
    assert patterns["all_fail"] == [2]
    assert patterns["adaptive_fails_any_static_succeeds"] == [3]
    counter = paired["counterexamples"]
    assert counter["adaptive_underperforms_light"] == [3]
    assert counter["adaptive_underperforms_strong"] == []
    assert sorted(counter["pattern_differs_from_reference"]) == [1, 2, 3]
    assert paired["paired_seed_count"] == 5


def test_fragility_flags_thin_battery_margins() -> None:
    records = paired_records()
    fragility = fragility_report(records, paired_comparison(records))
    assert fragility["adaptive_battery_margin"]["below_threshold_seeds"] == [4]
    assert any("battery" in flag for flag in fragility["flags"])
    single = fragility["adaptive_single_switch_successes"]
    assert single["count"] == single["of_successes"] == 3


def test_representative_selection_covers_the_interesting_cases() -> None:
    records = paired_records()
    chosen = select_representatives(paired_comparison(records), records)
    reasons = {entry["reason"]: entry["seed"] for entry in chosen}
    assert reasons["only_adaptive_succeeds"] == 0
    assert reasons["all_policies_fail"] == 2
    assert reasons["a_static_policy_succeeds"] in (1, 3)
    assert reasons["fragile_adaptive_success_battery_margin"] == 4
    assert reasons["adaptive_failure_counterexample"] == 3
    seeds = [entry["seed"] for entry in chosen]
    assert len(seeds) == len(set(seeds))  # no duplicates


def test_run_record_restates_the_mission_result() -> None:
    observations = [
        SimpleNamespace(
            execution={"energy_j": 600.0},
            score={"matched_target_ids": ["TGT_S1_WALKING_SMALL"]},
            action_valid=True,
        ),
        SimpleNamespace(
            execution={"energy_j": 10.0},
            score={"matched_target_ids": ["TGT_E2_STANDING_LATE"]},
            action_valid=False,
        ),
    ]
    result = SimpleNamespace(
        scenario_id="S_0001",
        policy_name=RULE,
        mission_success=True,
        constraints={
            "quality_success": True,
            "deadline_success": True,
            "battery_constraint_success": True,
            "communication_constraint_success": True,
        },
        termination_reason="path_complete",
        quality={
            "target_recall": 0.25,
            "detection_precision": 1.0,
            "false_positive_detections": 0,
            "false_positives_per_processed_minute": 0.0,
            "unique_targets_found": 2,
            "total_unique_targets": 8,
        },
        observations=observations,
        config_selection_history=("local_strong_real", "local_light_real"),
        final_battery_frac=0.25,
        cumulative_energy_j=6000.0,
        mean_executor_latency_s=0.9,
        processed_observation_count=2,
        skipped_observation_count=1,
    )
    scenario = SimpleNamespace(
        targets=[SimpleNamespace(object_id=f"TGT_S{i}_X_SMALL") for i in range(1, 5)]
        + [SimpleNamespace(object_id=f"TGT_E{i}_X_LATE") for i in range(1, 5)],
        contract=SimpleNamespace(min_final_battery_frac=0.22),
    )
    record = run_record(result, scenario, seed=1)
    assert record["compute_energy_j"] == pytest.approx(610.0)
    assert record["flight_energy_j"] == pytest.approx(5390.0)
    assert record["communication_energy_j"] is None
    assert record["battery_margin_frac"] == pytest.approx(0.03)
    assert record["config_switch_count"] == 1
    assert record["config_history_rle"] == [["local_strong_real", 1], ["local_light_real", 1]]
    assert record["invalid_action_count"] == record["fallback_count"] == 1
    assert record["small_targets_found"] == 1 and record["small_targets_total"] == 4
    assert record["late_targets_found"] == 1 and record["late_targets_total"] == 4
    assert not math.isnan(record["target_recall"])
