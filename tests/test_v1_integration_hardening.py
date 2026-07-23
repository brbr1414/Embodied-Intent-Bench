"""V1 integration hardening: model-agnostic fallback resolution, false-positive rate names,
and the frozen empirical schema versions.

These lock in the three interface fixes made when the empirical stack was integrated: a
benchmark no longer needs its catalog to contain CFG_LOCAL_LIGHT, the false-positive-rate
metrics are named for their denominators, and every externally consumed empirical schema
declares and validates a version.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from aerointentbench.benchmark import BenchmarkData, resolve_fallback_config_id, run_episode
from aerointentbench.metrics.episode_metrics import _with_mission_false_positive_rate
from aerointentbench.policies.static import StaticPolicy
from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from aerointentbench.schemas.contract import PrivacyLevel, load_contract
from aerointentbench.schemas.episode import Episode, load_episode
from aerointentbench.schemas.loading import SchemaValidationError, SchemaVersionError
from aerointentbench.tasks.base import TaskEvaluationResult

REPO = Path(__file__).resolve().parents[1]
EMP_ROOT = REPO / "data" / "examples" / "empirical_replay"


# --- fallback resolution helpers -------------------------------------------------------------


def cfg(config_id: str, placement: Placement = Placement.LOCAL) -> Configuration:
    return Configuration(config_id, "model", Strategy(placement, Precision.INT8))


def catalog(*ids: str) -> ConfigCatalog:
    return ConfigCatalog([cfg(i) for i in ids])


def episode(
    allowed: tuple[str, ...], *, fallback: str | None = None, initial: str | None = None
) -> Episode:
    return Episode(
        episode_id="E",
        platform_id="P",
        path_id="PA",
        frame_stream_id="S",
        initial_altitude_m=40.0,
        velocity_mps=5.0,
        initial_battery_frac=0.8,
        power_mode="15W",
        network_trace_id="T",
        allowed_config_ids=allowed,
        initial_config_id=initial,
        seed=0,
        fallback_config_id=fallback,
    )


def resolve(ep: Episode, cat: ConfigCatalog, *, explicit: str | None = None) -> str:
    return resolve_fallback_config_id(
        episode=ep, catalog=cat, privacy_level=PrivacyLevel.REMOTE_ALLOWED, explicit=explicit
    )


# --- fallback precedence (points 1-6) --------------------------------------------------------


def test_legacy_cfg_local_light_is_resolved_when_present_and_allowed() -> None:
    resolved = resolve(episode(("CFG_LOCAL_LIGHT", "CFG_X")), catalog("CFG_LOCAL_LIGHT", "CFG_X"))
    assert resolved == "CFG_LOCAL_LIGHT"  # step 4: legacy default, backward compatible


def test_episode_fallback_config_id_is_honoured() -> None:
    ep = episode(("CFG_A", "CFG_B"), fallback="CFG_B")
    assert resolve(ep, catalog("CFG_A", "CFG_B")) == "CFG_B"


def test_initial_config_id_is_used_when_no_explicit_or_episode_fallback() -> None:
    ep = episode(("CFG_A", "CFG_B"), initial="CFG_A")
    assert resolve(ep, catalog("CFG_A", "CFG_B")) == "CFG_A"  # step 3


def test_explicit_override_takes_precedence_over_episode_fallback() -> None:
    ep = episode(("CFG_A", "CFG_B"), fallback="CFG_B")
    assert resolve(ep, catalog("CFG_A", "CFG_B"), explicit="CFG_A") == "CFG_A"


def test_an_unknown_fallback_fails_before_execution() -> None:
    with pytest.raises(SchemaValidationError, match="unusable as a fallback"):
        resolve(episode(("CFG_A",)), catalog("CFG_A"), explicit="NOPE")


def test_a_disallowed_fallback_fails_before_execution() -> None:
    # CFG_B exists in the catalog but is not in the episode's allowed pool.
    with pytest.raises(SchemaValidationError, match="not in the episode's allowed pool"):
        resolve(episode(("CFG_A",)), catalog("CFG_A", "CFG_B"), explicit="CFG_B")


def test_no_usable_fallback_fails_requiring_an_explicit_one() -> None:
    with pytest.raises(SchemaValidationError, match="no usable safe fallback"):
        resolve(episode(("CFG_A",)), catalog("CFG_A"))  # no CFG_LOCAL_LIGHT, no episode fallback


def test_a_disallowed_episode_fallback_is_rejected_at_load(write_json) -> None:
    payload = {
        "schema_version": "1.0",
        "episode_id": "E",
        "platform_id": "P",
        "path_id": "PA",
        "frame_stream_id": "S",
        "initial_altitude_m": 40.0,
        "velocity_mps": 5.0,
        "initial_battery_frac": 0.8,
        "power_mode": "15W",
        "network_trace_id": "T",
        "allowed_config_ids": ["CFG_A"],
        "initial_config_id": None,
        "fallback_config_id": "CFG_B",
        "seed": 0,
    }
    with pytest.raises(SchemaValidationError, match=r"fallback_config_id .* is not in"):
        load_episode(write_json(payload))


# --- config-agnostic end to end (points 2, 7) ------------------------------------------------


def _emp_episode_contract():
    data = BenchmarkData(EMP_ROOT)
    ep = load_episode(EMP_ROOT / "episodes" / "episode.json")
    contract = load_contract(EMP_ROOT / "contracts" / "contract.json")
    return data, ep, contract


def test_a_bundle_without_cfg_local_light_runs_with_an_explicit_fallback() -> None:
    data, ep, contract = _emp_episode_contract()
    # An allowed pool that excludes CFG_LOCAL_LIGHT: the legacy default is unusable here.
    no_light = dataclasses.replace(ep, allowed_config_ids=("CFG_LOCAL_STRONG",))
    result = run_episode(
        data=data,
        episode=no_light,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_STRONG",
    )
    assert result.metrics.quality.details["quality_evaluation"] == "empirical_mask_iou"


def test_the_same_bundle_without_a_fallback_fails_before_running() -> None:
    data, ep, contract = _emp_episode_contract()
    no_light = dataclasses.replace(ep, allowed_config_ids=("CFG_LOCAL_STRONG",))
    with pytest.raises(SchemaValidationError, match="no usable safe fallback"):
        run_episode(
            data=data,
            episode=no_light,
            contract=contract,
            policy=StaticPolicy("CFG_LOCAL_STRONG"),
            executor_name="replay",
        )


def test_invalid_policy_actions_use_the_resolved_fallback() -> None:
    data, ep, contract = _emp_episode_contract()
    result = run_episode(
        data=data,
        episode=ep,
        contract=contract,
        policy=StaticPolicy("NOT_A_REAL_CONFIG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_STRONG",
    )
    record = result.record
    assert record.invalid_action_count == record.step_count > 0
    assert set(record.config_selection_history) == {"CFG_LOCAL_STRONG"}  # every action substituted


# --- false-positive rate names (points 8-10) -------------------------------------------------


def test_mission_false_positive_rate_uses_wall_clock_completion_time() -> None:
    data, ep, contract = _emp_episode_contract()
    result = run_episode(
        data=data,
        episode=ep,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_STRONG",
    )
    d = result.metrics.quality.details
    fp = d["false_positive_detections"]
    expected = fp * 60.0 / result.metrics.mission_completion_time_s
    assert d["false_positives_per_mission_minute"] == pytest.approx(expected)


def test_the_two_false_positive_rates_have_distinct_denominator_named_keys() -> None:
    data, ep, contract = _emp_episode_contract()
    result = run_episode(
        data=data,
        episode=ep,
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_STRONG"),
        executor_name="replay",
        fallback_config_id="CFG_LOCAL_STRONG",
    )
    d = result.metrics.quality.details
    assert "false_positives_per_processed_minute" in d
    assert "false_positives_per_mission_minute" in d
    assert "false_positives_per_minute" not in d  # the ambiguous name is gone


def test_zero_duration_mission_false_positive_rate_is_zero() -> None:
    evaluation = TaskEvaluationResult(
        metric_name="target_recall",
        value=1.0,
        operator=ComparisonOperator(">="),
        threshold=0.5,
        success=True,
        details={"false_positive_detections": 3},
    )
    augmented = _with_mission_false_positive_rate(evaluation, mission_time_s=0.0)
    assert augmented.details["false_positives_per_mission_minute"] == 0.0


def test_profile_evaluation_gains_no_false_positive_rate_field() -> None:
    # A precomputed/profile evaluation has no detection count, so it is left untouched.
    evaluation = TaskEvaluationResult(
        metric_name="target_f1",
        value=0.8,
        operator=ComparisonOperator(">="),
        threshold=0.8,
        success=True,
        details={"target_f1": 0.8},
    )
    assert _with_mission_false_positive_rate(evaluation, 900.0) is evaluation


# --- frozen empirical schema versions (points 11, 12) ----------------------------------------


def test_every_empirical_artifact_declares_schema_version_1_0() -> None:
    files = [
        EMP_ROOT / "ground_truth" / "gt_example_stream.json",
        EMP_ROOT / "predictions" / "empirical_replay.json",
        REPO / "data" / "examples" / "empirical_source" / "manifest.json",
        REPO / "data" / "examples" / "empirical_source" / "ground_truth.json",
    ]
    for path in files:
        assert json.loads(path.read_text())["schema_version"] == "1.0", path


def test_an_unsupported_empirical_schema_version_fails(write_json) -> None:
    from aerointentbench.tasks.human_search_segmentation import load_human_search_mask_ground_truth

    payload = json.loads((EMP_ROOT / "ground_truth" / "gt_example_stream.json").read_text())
    payload["schema_version"] = "2.0"
    with pytest.raises(SchemaVersionError):
        load_human_search_mask_ground_truth(write_json(payload))


def test_an_unsupported_manifest_schema_version_fails(write_json) -> None:
    from aerointentbench.tools.bundle_manifest import load_manifest

    payload = json.loads(
        (REPO / "data" / "examples" / "empirical_source" / "manifest.json").read_text()
    )
    payload["schema_version"] = "9.9"
    with pytest.raises(SchemaVersionError):
        load_manifest(write_json(payload))
