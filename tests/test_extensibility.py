"""The eight extensibility acceptance criteria from docs/architecture.md §8.

Each is stated there as "possible without modifying ``EpisodeRunner`` core logic". A claim
like that decays quietly: nothing fails when it stops being true, it just gets harder to add
the next thing. So each criterion gets a test that actually does the extension.

Adding a registry entry, a fixture, or a new module is allowed. Editing the runner is not --
guarded separately in ``test_episode_runner.py``, which parses the runner to confirm it
imports no task or policy module and never reads ``task_id``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from aerointentbench.benchmark import BenchmarkData, run_episode
from aerointentbench.executor import ReplayExecutor, ReplayRecord
from aerointentbench.registry import Registry
from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.contract import load_contract
from aerointentbench.schemas.network_trace import NetworkObservation
from aerointentbench.simulator.battery_model import BatteryState
from aerointentbench.simulator.network_trace import NetworkModel
from aerointentbench.tasks.base import TaskDefinition, TaskEvaluationResult
from aerointentbench.tasks.registry import task_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A writable copy of the shipped data, so a test can add fixtures to it."""
    root = tmp_path / "data"
    shutil.copytree(REPO_ROOT / "data", root)
    return root


@pytest.fixture
def contract():
    return load_contract(REPO_ROOT / "data" / "contracts" / "contract_001.json")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _episode(data: BenchmarkData, episode_id: str = "EPISODE_001"):
    return next(e for e in data.episodes() if e.episode_id == episode_id)


# --- 1. a new task evaluator ------------------------------------------------------------------


def test_criterion_1_a_new_task_is_a_registration_and_a_fixture(data_root: Path) -> None:
    """Add an object-detection task without touching the runner, metrics, or any policy."""

    class DetectionTracker:
        def __init__(self) -> None:
            self.frames = 0

        def update(self, result) -> None:
            self.frames += bool(result.success)

        def policy_summary(self):
            from aerointentbench.schemas.runtime_state import EvidenceSummary

            return EvidenceSummary(
                predicted_unique_targets=0,
                processed_frames=self.frames,
                mean_prediction_confidence=0.0,
            )

        def final_record(self):
            return self

        @property
        def task_id(self) -> str:
            return "OBJECT_DETECTION_TEST"

        @property
        def processed_frames(self) -> int:
            return self.frames

        def to_dict(self) -> dict[str, Any]:
            return {"task_id": self.task_id, "processed_frames": self.frames}

    class DetectionEvaluator:
        def evaluate(self, evidence, ground_truth, contract) -> TaskEvaluationResult:
            del ground_truth
            # A different metric name entirely: the point is that nothing downstream cares.
            return TaskEvaluationResult(
                metric_name="detection_ap",
                value=0.91,
                operator=ComparisonOperator.GREATER_EQUAL,
                threshold=contract.quality_threshold,
                success=contract.quality_threshold <= 0.91,
                details={"processed_frames": evidence.processed_frames},
            )

    class DetectionTask:
        def __init__(self, task_spec) -> None:
            self._spec = task_spec

        task_id = "OBJECT_DETECTION_TEST"

        @property
        def task_spec(self):
            return self._spec

        def create_tracker(self):
            return DetectionTracker()

        def create_evaluator(self):
            return DetectionEvaluator()

        def load_ground_truth(self, directory, episode):
            del directory, episode
            return None

        def create_prediction_source(self, ground_truth, profiles):
            del ground_truth, profiles
            return None

    _write(
        data_root / "task_specs" / "object_detection_test.json",
        {
            "schema_version": "1.0",
            "task_id": "OBJECT_DETECTION_TEST",
            "evidence_type": "box_set",
            "target_type": "vehicle",
            "ground_truth_type": "boxes",
            "matching_rule": {"metric": "box_iou", "operator": ">=", "threshold": 0.5},
            "deduplication": {"method": "ground_truth_track_id"},
            "supported_quality_metrics": ["detection_ap"],
        },
    )
    _write(
        data_root / "contracts" / "contract_detection.json",
        {
            "schema_version": "1.0",
            "contract_id": "CONTRACT_DETECTION",
            "task_id": "OBJECT_DETECTION_TEST",
            "evidence_type": "box_set",
            "quality_metric": "detection_ap",
            "quality_operator": ">=",
            "quality_threshold": 0.80,
            "deadline_s": 960.0,
            "communication_budget_mb": 400.0,
            "min_final_battery_frac": 0.20,
            "privacy_level": "remote_allowed",
        },
    )

    task_registry.register("OBJECT_DETECTION_TEST", DetectionTask)
    try:
        data = BenchmarkData(data_root)
        result = run_episode(
            data=data,
            episode=_episode(data),
            contract=load_contract(data_root / "contracts" / "contract_detection.json"),
            policy_name="always_local_light",
        )
    finally:
        # Registries have no unregister -- names are meant to be permanent within a process.
        # A test must not leave a fixture task behind for the rest of the session, so this
        # reaches in deliberately. Production code has no reason to.
        task_registry._factories.pop("OBJECT_DETECTION_TEST", None)

    assert result.metrics.quality.metric_name == "detection_ap"
    assert result.metrics.mission_success
    assert result.record.step_count == 900


# --- 2. a new policy --------------------------------------------------------------------------


def test_criterion_2_a_new_policy_is_a_class_conforming_to_the_protocol(
    data_root: Path, contract
) -> None:
    class AlternatingPolicy:
        def __init__(self) -> None:
            self._calls = 0

        def select_config(self, contract, state, configs) -> str:
            del contract, state
            self._calls += 1
            ids = configs.ids()
            return ids[self._calls % len(ids)]

    data = BenchmarkData(data_root)
    result = run_episode(
        data=data, episode=_episode(data), contract=contract, policy=AlternatingPolicy()
    )

    assert result.record.invalid_action_count == 0
    assert result.record.configuration_switch_count > 100


# --- 3. a different executor ------------------------------------------------------------------


def test_criterion_3_the_executor_is_swappable(data_root: Path, contract) -> None:
    data = BenchmarkData(data_root)
    episode = _episode(data)
    records = {
        (frame, "CFG_LOCAL_LIGHT"): ReplayRecord(
            frame_id=frame,
            config_id="CFG_LOCAL_LIGHT",
            success=True,
            latency_ms=100.0,
            onboard_energy_j=3.0,
            prediction=None,
        )
        for frame in range(901)
    }

    result = run_episode(
        data=data,
        episode=episode,
        contract=contract,
        policy_name="always_local_light",
        executor=ReplayExecutor(records),
        executor_name="replay",
    )

    assert result.record.executor_name == "replay"
    assert result.record.failed_inference_count == 0
    assert result.record.step_count == 900


# --- 4. a different battery model -------------------------------------------------------------


def test_criterion_4_the_battery_model_is_swappable(data_root: Path, contract) -> None:
    """The seam a recorded-discharge or electrochemical model plugs into."""

    class DoubleDrainBatteryModel:
        def transition(self, previous_state, usage, elapsed_time_s):
            del elapsed_time_s
            return BatteryState(
                capacity_j=previous_state.capacity_j,
                remaining_j=max(0.0, previous_state.remaining_j - 2.0 * usage.total_j),
            )

    data = BenchmarkData(data_root)
    episode = _episode(data)
    normal = run_episode(data=data, episode=episode, contract=contract)
    drained = run_episode(
        data=data, episode=episode, contract=contract, battery_model=DoubleDrainBatteryModel()
    )

    assert drained.metrics.final_battery_fraction < normal.metrics.final_battery_fraction

    # The replacement changes the run, not just the reported number: at twice the drain the
    # battery empties before the path is flown, so the episode ends early for a different
    # reason. Total energy is therefore *lower*, because fewer steps were taken -- the
    # ledger records what was spent, and less was spent.
    assert normal.record.termination_reason.value == "path_complete"
    assert drained.record.termination_reason.value == "battery_depleted"
    assert drained.record.step_count < normal.record.step_count
    assert drained.metrics.total_energy_j < normal.metrics.total_energy_j

    # Per-step accounting is untouched: only the battery's response to it changed.
    assert drained.record.steps[0].energy == normal.record.steps[0].energy


# --- 5. a new configuration strategy field ----------------------------------------------------


def test_criterion_5_a_new_strategy_dimension_needs_no_schema_change(
    data_root: Path, contract
) -> None:
    """A future dimension rides in ``strategy.parameters`` rather than a new typed field."""
    catalog_path = next((data_root / "configs").glob("*.json"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["configs"].append(
        {
            "config_id": "CFG_LOCAL_PRUNED",
            "model_id": "STRONG_INSTANCE_SEG",
            "strategy": {
                "placement": "local",
                "precision": "int8",
                "parameters": {"pruning_ratio": 0.4, "early_exit_index": 3},
            },
        }
    )
    _write(catalog_path, catalog)

    profile_path = next((data_root / "profiles").glob("*.json"))
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles["profiles"]["CFG_LOCAL_PRUNED"] = {
        "compute_latency_ms": 220.0,
        "onboard_energy_j": 6.0,
        "upload_mb": 0.0,
        "download_mb": 0.0,
        "quality_tier": "medium",
    }
    _write(profile_path, profiles)

    episode_path = data_root / "episodes" / "episode_001.json"
    episode_payload = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_payload["allowed_config_ids"].append("CFG_LOCAL_PRUNED")
    _write(episode_path, episode_payload)

    data = BenchmarkData(data_root)
    configuration = data.catalog.get("CFG_LOCAL_PRUNED")
    assert configuration.strategy.parameters["pruning_ratio"] == 0.4

    from aerointentbench.policies import StaticPolicy

    result = run_episode(
        data=data,
        episode=_episode(data),
        contract=contract,
        policy=StaticPolicy("CFG_LOCAL_PRUNED"),
    )
    assert result.record.invalid_action_count == 0
    assert result.record.steps[0].execution["latency_s"] == pytest.approx(0.22)


# --- 6. an optional task-specific metric ------------------------------------------------------


def test_criterion_6_a_task_metric_flows_through_untouched(data_root: Path, contract) -> None:
    """Extra task detail reaches the result without the metrics layer knowing about it."""
    data = BenchmarkData(data_root)
    result = run_episode(data=data, episode=_episode(data), contract=contract)

    details = result.metrics.quality.details
    assert {"target_precision", "target_recall", "matched_targets"} <= set(details)
    # Reported, but taking no part in deciding success.
    assert result.metrics.mission_success == result.metrics.quality.success
    assert "target_precision" in json.dumps(result.metrics.to_dict())


# --- 7. a new network trace -------------------------------------------------------------------


def test_criterion_7_a_new_trace_is_a_file(data_root: Path, contract) -> None:
    _write(
        data_root / "network_traces" / "synthetic_network_flapping_001.json",
        {
            "schema_version": "1.0",
            "trace_id": "NETWORK_FLAPPING_001",
            "segments": [
                {
                    "start_s": start,
                    "end_s": start + 120,
                    "bandwidth_mbps": 20.0 if (start // 120) % 2 == 0 else 0.0,
                    "rtt_ms": 30.0 if (start // 120) % 2 == 0 else 0.0,
                    "packet_loss_frac": 0.0 if (start // 120) % 2 == 0 else 1.0,
                }
                for start in range(0, 960, 120)
            ],
        },
    )
    episode_path = data_root / "episodes" / "episode_001.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["network_trace_id"] = "NETWORK_FLAPPING_001"
    _write(episode_path, payload)

    data = BenchmarkData(data_root)
    result = run_episode(
        data=data,
        episode=_episode(data),
        contract=contract,
        policy_name="always_remote_strong",
    )

    assert result.record.failed_inference_count > 0, "the dead windows must bite"
    assert len(result.record.network_change_times_s) == 7


# --- 8. an external simulator adapter ---------------------------------------------------------


def test_criterion_8_an_external_network_source_plugs_in(data_root: Path, contract) -> None:
    """Where a Gazebo, PX4, or hardware-in-the-loop adapter would attach.

    Such an adapter supplies observations from outside; conforming to ``NetworkModel`` is the
    whole requirement, and the runner cannot tell the difference.
    """

    class ScriptedExternalLink:
        """Stands in for an adapter reading from an external simulator."""

        def __init__(self) -> None:
            self.queries: list[float] = []

        def observe(self, time_s: float) -> NetworkObservation:
            self.queries.append(time_s)
            usable = (time_s // 100) % 2 == 0
            return NetworkObservation(
                bandwidth_mbps=25.0 if usable else 0.0,
                rtt_ms=20.0 if usable else 0.0,
                packet_loss_frac=0.0 if usable else 1.0,
            )

    adapter: NetworkModel = ScriptedExternalLink()
    data = BenchmarkData(data_root)
    result = run_episode(
        data=data,
        episode=_episode(data),
        contract=contract,
        policy_name="rule_based",
        network_model=adapter,
    )

    assert adapter.queries, "the runner drove the adapter"
    assert len(adapter.queries) == result.record.step_count
    # The adapter exposes no change_times_s, and the runner tolerates that.
    assert result.record.network_change_times_s == ()


# --- the registry mechanism itself ------------------------------------------------------------


def test_a_registry_keeps_implementations_isolated() -> None:
    scratch: Registry[object] = Registry("thing")
    scratch.register("a", lambda **_: "A")
    assert scratch.create("a") == "A"
    assert "a" not in task_registry


def test_the_task_protocol_is_satisfied_by_the_shipped_task() -> None:
    from aerointentbench.schemas.task_spec import load_task_spec
    from aerointentbench.tasks.registry import resolve_task

    spec = load_task_spec(REPO_ROOT / "data" / "task_specs" / "human_search_segmentation.json")
    task: TaskDefinition = resolve_task(spec)

    for method in (
        "create_tracker",
        "create_evaluator",
        "load_ground_truth",
        "create_prediction_source",
    ):
        assert callable(getattr(task, method))
