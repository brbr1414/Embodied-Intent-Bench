"""Periodic ground-station telemetry: uplink traffic independent of remote inference.

A search UAV reports mission progress to its ground station even when perception is
fully local. These tests pin the semantics: reports are scheduled on mission time and
attempted against the capture-time network sample; sent reports share the contract's
communication budget and charge transmit energy to the battery; reports during an
outage are lost and cost nothing; a scenario without the ``telemetry`` field produces
exactly the traffic-free behaviour it always had. CI-safe: tiny PNG world, heuristic
executors, no Torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.runner import MissionRunner
from aerointentbench.v2.scenario import load_scenario
from tests.test_v2_visual_loop import scenario_payload, write_world_png

#: 1 Hz status reports; the mission's polyline runs 120 m at 5 m/s -> 24 s -> 24 reports.
TELEMETRY = {"interval_s": 1.0, "report_mb": 0.1, "energy_j_per_mb": 10.0, "max_loss_frac": 0.5}


def _run(tmp_path: Path, **overrides):
    world = write_world_png(tmp_path / "world.png")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario_payload(world, **overrides)))
    return MissionRunner(load_scenario(path), "always_fast").run()


class TestAbsentField:
    def test_no_telemetry_means_no_traffic_and_no_new_costs(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        assert result.telemetry_reports_sent == 0
        assert result.telemetry_reports_lost == 0
        assert result.telemetry_mb == 0.0
        assert result.telemetry_energy_j == 0.0
        assert result.cumulative_communication_mb == 0.0


class TestAccounting:
    def test_reports_size_energy_and_shared_ledgers(self, tmp_path: Path) -> None:
        base = _run(tmp_path)
        result = _run(tmp_path, **{"simulation.telemetry": TELEMETRY})
        assert result.telemetry_reports_sent == 24
        assert result.telemetry_reports_lost == 0
        assert result.telemetry_mb == pytest.approx(2.4)
        assert result.telemetry_energy_j == pytest.approx(24.0)
        # Sent MB share the mission communication ledger; energy reaches the battery.
        assert result.cumulative_communication_mb == pytest.approx(
            base.cumulative_communication_mb + 2.4
        )
        assert result.cumulative_energy_j == pytest.approx(base.cumulative_energy_j + 24.0)
        # Remote-transfer energy stays a separate ledger.
        assert result.communication_energy_j == base.communication_energy_j
        payload = result.to_dict(include_observations=False)
        assert payload["telemetry_reports_sent"] == 24
        assert payload["telemetry_mb"] == pytest.approx(2.4)

    def test_perception_is_untouched(self, tmp_path: Path) -> None:
        base = _run(tmp_path)
        result = _run(tmp_path, **{"simulation.telemetry": TELEMETRY})
        assert result.quality == base.quality
        assert result.config_selection_history == base.config_selection_history
        assert result.final_time_s == base.final_time_s
        assert result.skipped_observation_ids == base.skipped_observation_ids


class TestLinkInteraction:
    def test_reports_during_an_outage_are_lost_and_free(self, tmp_path: Path) -> None:
        trace = [
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
        result = _run(
            tmp_path,
            **{"simulation.telemetry": TELEMETRY, "simulation.network_trace": trace},
        )
        # Reports at t=11..20 fall in the outage; t=1..10 and t=21..24 go through.
        assert result.telemetry_reports_lost == 10
        assert result.telemetry_reports_sent == 14
        assert result.telemetry_mb == pytest.approx(1.4)
        assert result.telemetry_energy_j == pytest.approx(14.0)

    def test_loss_above_the_configured_bound_drops_reports(self, tmp_path: Path) -> None:
        strict = dict(TELEMETRY, max_loss_frac=0.1)
        result = _run(
            tmp_path,
            **{
                "simulation.telemetry": strict,
                "simulation.network": {
                    "bandwidth_mbps": 20.0,
                    "rtt_ms": 30.0,
                    "packet_loss_frac": 0.3,
                },
            },
        )
        assert result.telemetry_reports_sent == 0
        assert result.telemetry_reports_lost == 24
        assert result.telemetry_mb == 0.0


class TestBudgetCoupling:
    def test_telemetry_alone_can_exhaust_the_communication_budget(self, tmp_path: Path) -> None:
        heavy = dict(TELEMETRY, report_mb=1.0)
        result = _run(tmp_path, **{"simulation.telemetry": heavy})
        assert result.telemetry_mb == pytest.approx(24.0)  # budget is 10 MB
        assert result.constraints["communication_constraint_success"] is False
        assert result.mission_success is False


class TestSchema:
    def test_unknown_telemetry_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError):
            _run(tmp_path, **{"simulation.telemetry": dict(TELEMETRY, split_ratio=0.5)})

    def test_non_positive_interval_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError):
            _run(tmp_path, **{"simulation.telemetry": dict(TELEMETRY, interval_s=0.0)})
