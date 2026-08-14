"""Policy-execution cost: the decision-maker is a resource consumer too.

The benchmark historically charged nothing for running the selection policy. The
``simulation.policy_execution`` block makes each decision cost what its deployment
costs: onboard decisions spend mission clock and battery (a slow policy skips capture
slots exactly like a slow executor); server-hosted decisions are a state-up/action-down
round trip that shares the communication budget, derives its latency from the
capture-time network sample, and is LOST during an outage — the drone then acts on the
declared ``on_lost_decision`` rule and the policy is genuinely never consulted. A
scenario without the block keeps the historical free-policy behaviour. CI-safe: tiny
PNG world, heuristic executors, no Torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.runner import MissionRunner
from aerointentbench.v2.scenario import load_scenario
from tests.test_v2_visual_loop import scenario_payload, write_world_png

ONBOARD_FREE = {"location": "onboard"}
ONBOARD_COSTLY = {
    "location": "onboard",
    "latency_s_per_decision": 0.8,
    "energy_j_per_decision": 2.0,
}
SERVER = {
    "location": "server",
    "state_mb_per_decision": 0.01,
    "action_mb_per_decision": 0.001,
    "server_compute_s": 0.05,
    "energy_j_per_mb": 10.0,
    "max_loss_frac": 0.5,
}
OUTAGE_TRACE = [
    {
        "regime_id": "good",
        "start_s": 0.0,
        "uplink_mbps": 100.0,
        "downlink_mbps": 200.0,
        "rtt_ms": 20.0,
        "packet_loss_frac": 0.0,
    },
    {
        "regime_id": "disconnected",
        "start_s": 10.5,
        "uplink_mbps": 0.0,
        "downlink_mbps": 0.0,
        "rtt_ms": 0.0,
        "packet_loss_frac": 1.0,
    },
    {
        "regime_id": "recovered",
        "start_s": 20.5,
        "uplink_mbps": 50.0,
        "downlink_mbps": 100.0,
        "rtt_ms": 30.0,
        "packet_loss_frac": 0.0,
    },
]
DEAD_TRACE = [
    {
        "regime_id": "disconnected",
        "start_s": 0.0,
        "uplink_mbps": 0.0,
        "downlink_mbps": 0.0,
        "rtt_ms": 0.0,
        "packet_loss_frac": 1.0,
    }
]


def _run(tmp_path: Path, policy: str = "always_fast", **overrides):
    world = write_world_png(tmp_path / "world.png")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario_payload(world, **overrides)))
    return MissionRunner(load_scenario(path), policy).run()


class TestAbsentField:
    def test_no_block_means_the_historical_free_policy(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        assert result.policy_execution_location is None
        assert result.policy_decisions_made == 0
        assert result.policy_decisions_lost == 0
        assert result.policy_decision_time_s == 0.0
        assert result.policy_decision_energy_j == 0.0
        assert result.policy_decision_mb == 0.0
        assert all(log.policy_decision is None for log in result.observations)


class TestOnboard:
    def test_zero_cost_onboard_changes_nothing_but_the_ledger(self, tmp_path: Path) -> None:
        base = _run(tmp_path)
        result = _run(tmp_path, **{"simulation.policy_execution": ONBOARD_FREE})
        assert result.policy_execution_location == "onboard"
        assert result.policy_decisions_made == result.processed_observation_count
        assert result.policy_decision_time_s == 0.0
        assert result.policy_decision_energy_j == 0.0
        assert result.final_time_s == base.final_time_s
        assert result.cumulative_energy_j == pytest.approx(base.cumulative_energy_j)
        assert result.config_selection_history == base.config_selection_history
        assert result.skipped_observation_ids == base.skipped_observation_ids

    def test_decision_energy_reaches_the_battery(self, tmp_path: Path) -> None:
        result = _run(tmp_path, **{"simulation.policy_execution": ONBOARD_COSTLY})
        made = result.policy_decisions_made
        assert made > 0
        assert result.policy_decision_energy_j == pytest.approx(2.0 * made)
        assert result.policy_decision_time_s == pytest.approx(0.8 * made)

    def test_a_slow_policy_skips_slots_like_a_slow_executor(self, tmp_path: Path) -> None:
        base = _run(tmp_path)
        result = _run(tmp_path, **{"simulation.policy_execution": ONBOARD_COSTLY})
        # FAST executor is 0.4 s; +0.8 s of decision pushes each slot past the 1 s
        # cadence, so every observation now skips the next capture.
        assert base.skipped_observation_count == 0
        assert result.skipped_observation_count > 0
        assert result.processed_observation_count < base.processed_observation_count


class TestServer:
    def test_round_trip_shares_the_communication_budget(self, tmp_path: Path) -> None:
        base = _run(tmp_path)
        result = _run(tmp_path, **{"simulation.policy_execution": SERVER})
        made = result.policy_decisions_made
        assert made == result.processed_observation_count
        assert result.policy_decisions_lost == 0
        round_mb = 0.011
        assert result.policy_decision_mb == pytest.approx(round_mb * made)
        assert result.cumulative_communication_mb == pytest.approx(
            base.cumulative_communication_mb + round_mb * made
        )
        assert result.policy_decision_energy_j == pytest.approx(round_mb * made * 10.0)
        assert result.cumulative_energy_j > base.cumulative_energy_j

    def test_decision_latency_is_derived_from_the_network_sample(self, tmp_path: Path) -> None:
        result = _run(tmp_path, **{"simulation.policy_execution": SERVER})
        # Constant network: 20 Mbps both ways, RTT 30 ms.
        expected = 0.01 * 8.0 / 20.0 + 0.030 + 0.05 + 0.001 * 8.0 / 20.0
        made = result.policy_decisions_made
        assert result.policy_decision_time_s == pytest.approx(expected * made)
        for log in result.observations:
            assert log.policy_decision is not None
            assert log.policy_decision["lost"] is False
            assert log.policy_decision["latency_s"] == pytest.approx(expected)
            assert log.completion_time_s == pytest.approx(
                log.capture_time_s + expected + log.execution["mission_latency_s"]
            )

    def test_decisions_during_an_outage_are_lost_and_free(self, tmp_path: Path) -> None:
        result = _run(
            tmp_path,
            **{
                "simulation.policy_execution": SERVER,
                "simulation.network_trace": OUTAGE_TRACE,
            },
        )
        lost_logs = [log for log in result.observations if log.policy_decision["lost"]]
        assert result.policy_decisions_lost == len(lost_logs) > 0
        assert result.policy_decisions_made + result.policy_decisions_lost == len(
            result.observations
        )
        # Lost decisions transferred nothing: MB and energy count only made decisions.
        assert result.policy_decision_mb == pytest.approx(0.011 * result.policy_decisions_made)


class TestLostDecisionRules:
    """During an outage the drone acts on the declared rule — the policy is not consulted."""

    HOLD: ClassVar[dict] = dict(SERVER, on_lost_decision="hold")
    FALLBACK: ClassVar[dict] = dict(SERVER, on_lost_decision="fallback")

    def test_hold_keeps_the_current_configuration(self, tmp_path: Path) -> None:
        result = _run(
            tmp_path,
            policy="STRONG",
            **{
                "simulation.policy_execution": self.HOLD,
                "simulation.network_trace": OUTAGE_TRACE,
            },
        )
        assert result.policy_decisions_lost > 0
        for log in result.observations:
            if log.policy_decision["lost"]:
                assert log.executed_config_id == "STRONG"

    def test_fallback_drops_to_the_scenario_fallback(self, tmp_path: Path) -> None:
        result = _run(
            tmp_path,
            policy="STRONG",
            **{
                "simulation.policy_execution": self.FALLBACK,
                "simulation.network_trace": OUTAGE_TRACE,
            },
        )
        assert result.policy_decisions_lost > 0
        for log in result.observations:
            if log.policy_decision["lost"]:
                assert log.executed_config_id == "FAST"

    def test_a_dead_link_blinds_a_server_policy_for_the_whole_mission(self, tmp_path: Path) -> None:
        # The static STRONG policy would pin STRONG on every slot — but the server
        # never hears from it, so the mission runs entirely on the fallback config.
        result = _run(
            tmp_path,
            policy="STRONG",
            **{
                "simulation.policy_execution": self.HOLD,
                "simulation.network_trace": DEAD_TRACE,
            },
        )
        assert result.policy_decisions_made == 0
        assert result.policy_decisions_lost == len(result.observations) > 0
        assert set(result.config_selection_history) == {"FAST"}
        assert result.policy_decision_mb == 0.0
        assert result.policy_decision_energy_j == 0.0


class TestSchema:
    def test_server_policy_is_forbidden_under_local_only(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError, match="local_only"):
            _run(
                tmp_path,
                **{
                    "simulation.policy_execution": SERVER,
                    "mission_contract.privacy_level": "local_only",
                },
            )

    def test_onboard_policy_is_legal_under_local_only(self, tmp_path: Path) -> None:
        result = _run(
            tmp_path,
            **{
                "simulation.policy_execution": ONBOARD_FREE,
                "mission_contract.privacy_level": "local_only",
            },
        )
        assert result.policy_execution_location == "onboard"

    def test_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError):
            _run(
                tmp_path,
                **{"simulation.policy_execution": dict(ONBOARD_FREE, gpu_boost=True)},
            )

    def test_cross_location_fields_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError, match="state_mb_per_decision"):
            _run(
                tmp_path,
                **{"simulation.policy_execution": dict(ONBOARD_FREE, state_mb_per_decision=0.01)},
            )
        with pytest.raises(SchemaValidationError, match="latency_s_per_decision"):
            _run(
                tmp_path,
                **{"simulation.policy_execution": dict(SERVER, latency_s_per_decision=0.1)},
            )

    def test_bad_location_and_bad_rule_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError, match="location"):
            _run(tmp_path, **{"simulation.policy_execution": {"location": "cloud"}})
        with pytest.raises(SchemaValidationError, match="on_lost_decision"):
            _run(
                tmp_path,
                **{"simulation.policy_execution": dict(SERVER, on_lost_decision="panic")},
            )
