"""V3 P3: the GT-aware offline skyline and the budget-planner baseline.

CI-safe: tiny PNG worlds and the heuristic executors only — no rasters, no assets,
no Torch. The skyline's defining property under test is *soundness as an upper
bound*: no legal policy run may ever beat it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="V3 tests need the [v2] extras")

from aerointentbench.policies.budget_planner import (
    BudgetPlannerPolicy,
    BudgetPlannerSettings,
)
from aerointentbench.policies.registry import policy_registry
from aerointentbench.schemas.network_trace import NetworkObservation
from aerointentbench.schemas.profile import (
    PublicProfile,
    PublicProfileView,
    QualityTier,
)
from aerointentbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from aerointentbench.v2.runner import run_mission
from aerointentbench.v2.scenario import load_scenario
from aerointentbench.v2.skyline import compute_skyline
from tests.test_v2_remote_network import REMOTE_CONFIG, TRACE
from tests.test_v2_visual_loop import scenario_payload, write_world_png

# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture
def make_scenario(tmp_path: Path):
    world = write_world_png(tmp_path / "world.png")

    def _make(*, remote: bool = False, **overrides) -> Path:
        payload = scenario_payload(world)
        if remote:
            payload["executor_configs"].append(json.loads(json.dumps(REMOTE_CONFIG)))
            payload["simulation"]["network_trace"] = json.loads(json.dumps(TRACE))
        for dotted, value in overrides.items():
            node = payload
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _make


# --- skyline soundness -----------------------------------------------------------------------


def test_skyline_is_an_upper_bound_over_every_policy(make_scenario) -> None:
    scenario = load_scenario(make_scenario(remote=True))
    sky = compute_skyline(scenario)
    for policy in ("always_fast", "always_strong", "rule_based", "budget_planner"):
        result = run_mission(scenario, policy)
        assert sky.best_recall >= result.quality["target_recall"] - 1e-9, policy
        if result.mission_success:
            assert sky.best_success, policy


def test_skyline_is_deterministic(make_scenario) -> None:
    scenario = load_scenario(make_scenario(remote=True))
    a = compute_skyline(scenario).to_dict()
    b = compute_skyline(scenario).to_dict()
    assert a == b


def test_skyline_is_labelled_gt_aware(make_scenario) -> None:
    payload = compute_skyline(load_scenario(make_scenario())).to_dict()["skyline"]
    assert payload["gt_aware"] is True
    assert "UPPER BOUND" in payload["label"]
    assert "not a policy" in payload["label"]


def test_skyline_respects_privacy(make_scenario) -> None:
    scenario = load_scenario(
        make_scenario(remote=True, **{"mission_contract.privacy_level": "local_only"})
    )
    sky = compute_skyline(scenario)
    assert "REMOTE_STRONG" not in sky.legal_config_ids
    assert all(config != "REMOTE_STRONG" for _slot, config in sky.best_sequence)


def test_skyline_respects_the_communication_budget(make_scenario) -> None:
    scenario = load_scenario(
        make_scenario(remote=True, **{"mission_contract.communication_budget_mb": 3.0})
    )
    sky = compute_skyline(scenario)
    assert sky.best_communication_mb <= 3.0 + 1e-9


def test_skyline_rejects_non_recall_contracts(make_scenario) -> None:
    scenario = load_scenario(
        make_scenario(**{"mission_contract.quality_metric": "detection_precision"})
    )
    with pytest.raises(ValueError, match="target_recall"):
        compute_skyline(scenario)


def test_skyline_sequence_replays_to_its_own_claim(make_scenario) -> None:
    """Running the skyline's chosen sequence as a static schedule reproduces recall."""
    scenario_path = make_scenario(remote=True)
    scenario = load_scenario(scenario_path)
    sky = compute_skyline(scenario)
    schedule = dict(sky.best_sequence)

    class _SchedulePolicy:
        def select_config(self, contract, state, configs):
            return schedule.get(state.frame_id, next(iter(configs)).config_id)

    from aerointentbench.v2.runner import MissionRunner

    runner = MissionRunner(scenario, next(iter(schedule.values())))
    runner._policy = _SchedulePolicy()  # inject: the schedule is the point of the test
    result = runner.run()
    assert result.quality["target_recall"] == pytest.approx(sky.best_recall)


# --- budget planner (V1-interface, CI-safe) ---------------------------------------------------


def _profiles() -> PublicProfileView:
    return PublicProfileView(
        {
            "FAST": PublicProfile(
                config_id="FAST",
                expected_latency_ms=400.0,
                expected_upload_mb=0.0,
                quality_tier=QualityTier("low"),
            ),
            "REMOTE": PublicProfile(
                config_id="REMOTE",
                expected_latency_ms=300.0,
                expected_upload_mb=2.0,
                quality_tier=QualityTier("high"),
            ),
        }
    )


def _state(
    *,
    time_s: float,
    battery: float,
    comm: float,
    progress: float,
    bandwidth: float = 50.0,
) -> RuntimeState:
    return RuntimeState(
        current_time_s=time_s,
        frame_id=int(time_s),
        battery_frac=battery,
        power_mode="test",
        network=NetworkObservation(bandwidth_mbps=bandwidth, rtt_ms=20.0, packet_loss_frac=0.0),
        current_config_id=None,
        remaining_deadline_s=60.0 - time_s,
        cumulative_energy_j=0.0,
        cumulative_communication_mb=comm,
        path_progress=progress,
        evidence_summary=EvidenceSummary(
            predicted_unique_targets=0, processed_frames=0, mean_prediction_confidence=0.0
        ),
    )


def test_budget_planner_is_registered() -> None:
    assert "budget_planner" in policy_registry.names()


def test_planner_paces_the_communication_budget(v1_catalog_local_remote) -> None:
    """Early in the mission a nearly-spent budget forbids the upload-heavy config."""
    catalog, contract = v1_catalog_local_remote
    policy = BudgetPlannerPolicy(public_profiles=_profiles())
    # 10% through the mission with 90% of the budget gone: the per-step allowance is
    # far below the remote upload, so pacing must choose the local config.
    state = _state(
        time_s=5.0, battery=0.9, comm=0.9 * contract.communication_budget_mb, progress=0.1
    )
    assert policy.select_config(contract, state, catalog) == "FAST"
    # A fresh budget affords the upload: the planner takes the higher tier.
    fresh = _state(time_s=5.0, battery=0.9, comm=0.0, progress=0.1)
    planner = BudgetPlannerPolicy(public_profiles=_profiles())
    assert planner.select_config(contract, fresh, catalog) == "REMOTE"


def test_planner_projects_battery_from_observed_drain(v1_catalog_local_remote) -> None:
    catalog, contract = v1_catalog_local_remote
    policy = BudgetPlannerPolicy(
        public_profiles=_profiles(), settings=BudgetPlannerSettings(battery_margin_frac=0.03)
    )
    # Two observations establish a steep drain: 2%/s at 30% progress means the
    # projected final battery is far below the floor -> the fast path must win even
    # though the instantaneous battery (0.56) is nowhere near the floor.
    policy.select_config(
        contract, _state(time_s=10.0, battery=0.60, comm=0.0, progress=0.25), catalog
    )
    chosen = policy.select_config(
        contract, _state(time_s=12.0, battery=0.56, comm=0.0, progress=0.30), catalog
    )
    assert chosen == "FAST"


def test_planner_without_history_does_not_project(v1_catalog_local_remote) -> None:
    """The very first decision has no drain sample; it must not panic-switch."""
    catalog, contract = v1_catalog_local_remote
    policy = BudgetPlannerPolicy(public_profiles=_profiles())
    chosen = policy.select_config(
        contract, _state(time_s=0.0, battery=1.0, comm=0.0, progress=0.0), catalog
    )
    assert chosen == "REMOTE"


def test_planner_reset_clears_the_drain_estimate(v1_catalog_local_remote) -> None:
    catalog, contract = v1_catalog_local_remote
    policy = BudgetPlannerPolicy(public_profiles=_profiles())
    policy.select_config(
        contract, _state(time_s=10.0, battery=0.60, comm=0.0, progress=0.25), catalog
    )
    policy.select_config(
        contract, _state(time_s=12.0, battery=0.56, comm=0.0, progress=0.30), catalog
    )
    policy.reset()
    chosen = policy.select_config(
        contract, _state(time_s=0.0, battery=1.0, comm=0.0, progress=0.0), catalog
    )
    assert chosen == "REMOTE"


@pytest.fixture
def v1_catalog_local_remote():
    """A two-config V1 catalog (local FAST, remote REMOTE) plus a matching contract."""
    from aerointentbench.schemas.common import ComparisonOperator
    from aerointentbench.schemas.configuration import (
        ConfigCatalog,
        Configuration,
        Placement,
        Precision,
        Strategy,
    )
    from aerointentbench.schemas.contract import Contract, PrivacyLevel

    catalog = ConfigCatalog(
        (
            Configuration(
                config_id="FAST",
                model_id="M_FAST",
                strategy=Strategy(placement=Placement.LOCAL, precision=Precision.FP32),
            ),
            Configuration(
                config_id="REMOTE",
                model_id="M_REMOTE",
                strategy=Strategy(
                    placement=Placement.REMOTE,
                    precision=Precision.FP32,
                    parameters={"transmitted_payload": "raw_input"},
                ),
            ),
        ),
        catalog_id="V3_TEST_CATALOG",
    )
    contract = Contract(
        contract_id="V3_TEST_CONTRACT",
        task_id="human_search",
        evidence_type="detection",
        quality_metric="target_recall",
        quality_operator=ComparisonOperator.GREATER_EQUAL,
        quality_threshold=0.5,
        deadline_s=60.0,
        communication_budget_mb=20.0,
        min_final_battery_frac=0.2,
        privacy_level=PrivacyLevel("remote_allowed"),
    )
    return catalog, contract
