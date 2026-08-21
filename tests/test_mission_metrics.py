"""Numeric mission metrics: margins restate the evaluator, regret restates the skyline.

Pinned: each margin's sign reproduces the evaluator's constraint verdict (and the
verifier fails loudly when it would not); min_margin >= 0 coincides with
mission_success; the axis math matches hand computation for >= and <= contracts;
strict operators are a loud error at the metric layer and an `unavailable` note at
the runner layer; skyline regret is non-negative for a real mission and refuses
non-target_recall metrics. CI-safe: tiny PNG world, heuristic executors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.contract import Contract, PrivacyLevel
from aerointentbench.v2.mission_metrics import (
    contract_margins,
    skyline_regret,
    verify_margins,
)
from aerointentbench.v2.runner import MissionRunner
from aerointentbench.v2.scenario import load_scenario
from tests.test_v2_visual_loop import scenario_payload, write_world_png


def _contract(operator=ComparisonOperator.GREATER_EQUAL, threshold=0.5) -> Contract:
    return Contract(
        contract_id="M_TEST",
        task_id="HUMAN_SEARCH_SEGMENTATION",
        evidence_type="instance_mask_set",
        quality_metric="target_recall",
        quality_operator=operator,
        quality_threshold=threshold,
        deadline_s=60.0,
        communication_budget_mb=10.0,
        min_final_battery_frac=0.2,
        privacy_level=PrivacyLevel.REMOTE_ALLOWED,
    )


def _margins(contract, **overrides):
    values = {
        "quality_value": 0.75,
        "final_time_s": 45.0,
        "final_battery_frac": 0.6,
        "communication_mb": 2.5,
        "privacy_violation_count": 0,
        "processed_observation_count": 40,
    }
    values.update(overrides)
    return contract_margins(contract, **values)


class TestAxisMath:
    def test_hand_computed_margins(self) -> None:
        report = _margins(_contract())
        assert report.margins["quality"] == pytest.approx((0.75 - 0.5) / 0.5)
        assert report.margins["deadline"] == pytest.approx(15.0 / 60.0)
        assert report.margins["battery"] == pytest.approx((0.6 - 0.2) / 0.8)
        assert report.margins["communication"] == pytest.approx(7.5 / 10.0)
        assert report.margins["privacy"] == 0.0
        # Sanity axes (privacy: binary; deadline: path-determined on fixed-path
        # missions) do not participate while satisfied; the tightest
        # policy-sensitive axis binds.
        assert report.min_margin == pytest.approx(0.5)
        assert report.binding_axis in ("quality", "battery")

    def test_less_equal_contract_mirrors(self) -> None:
        contract = _contract(operator=ComparisonOperator.LESS_EQUAL, threshold=0.4)
        report = _margins(contract, quality_value=0.1)
        assert report.margins["quality"] == pytest.approx((0.4 - 0.1) / 0.4)

    def test_violation_makes_privacy_margin_negative(self) -> None:
        report = _margins(_contract(), privacy_violation_count=4)
        assert report.margins["privacy"] == pytest.approx(-0.1)
        assert report.min_margin == pytest.approx(-0.1)
        assert report.binding_axis == "privacy"

    def test_strict_operator_is_a_loud_error(self) -> None:
        with pytest.raises(ValueError, match="non-strict"):
            _margins(_contract(operator=ComparisonOperator.GREATER))


class TestVerification:
    def test_sign_divergence_fails_loudly(self) -> None:
        report = _margins(_contract())
        good = {
            "quality_success": True,
            "deadline_success": True,
            "battery_constraint_success": True,
            "communication_constraint_success": True,
            "privacy_constraint_success": True,
        }
        verify_margins(report, good)  # sanity: agreement passes
        with pytest.raises(AssertionError, match="battery"):
            verify_margins(report, dict(good, battery_constraint_success=False))


class TestRunnerIntegration:
    def _run(self, tmp_path: Path, **overrides):
        world = write_world_png(tmp_path / "world.png")
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world, **overrides)))
        return MissionRunner(load_scenario(path), "always_strong").run()

    def test_margins_ship_with_the_result_and_match_success(self, tmp_path: Path) -> None:
        result = self._run(tmp_path)
        payload = result.contract_margins
        assert set(payload["margins"]) == {
            "quality",
            "deadline",
            "battery",
            "communication",
            "privacy",
        }
        assert (payload["min_margin"] >= 0.0) == result.mission_success
        assert payload["margins"][payload["binding_axis"]] == payload["min_margin"]
        assert result.to_dict(include_observations=False)["contract_margins"] == payload

    def test_failing_mission_has_negative_min_margin(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, **{"mission_contract.min_final_battery_frac": 0.99})
        assert not result.mission_success
        assert result.contract_margins["min_margin"] < 0.0
        assert result.contract_margins["binding_axis"] == "battery"

    def test_violated_deadline_still_forces_the_minimum_negative(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, **{"mission_contract.deadline_s": 5.0})
        assert not result.constraints["deadline_success"]
        assert result.contract_margins["min_margin"] < 0.0

    def test_strict_operator_records_unavailable(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, **{"mission_contract.quality_operator": ">"})
        assert "unavailable" in result.contract_margins


class TestSkylineRegret:
    def test_regret_is_nonnegative_for_a_real_mission(self, tmp_path: Path) -> None:
        from aerointentbench.v2.skyline import compute_skyline

        world = write_world_png(tmp_path / "world.png")
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world)))
        scenario = load_scenario(path)
        sky = compute_skyline(scenario)
        result = MissionRunner(scenario, "always_fast").run()
        regret = skyline_regret(sky.best_recall, result)
        assert regret >= 0.0

    def test_non_recall_metric_is_refused(self, tmp_path: Path) -> None:
        world = write_world_png(tmp_path / "world.png")
        path = tmp_path / "scenario.json"
        overrides = {"mission_contract.quality_metric": "detection_precision"}
        path.write_text(json.dumps(scenario_payload(world, **overrides)))
        result = MissionRunner(load_scenario(path), "always_fast").run()
        with pytest.raises(ValueError, match="target_recall"):
            skyline_regret(1.0, result)


class TestOperatorTimeliness:
    """Per-target delivery ledger: visible -> matched -> delivered, evidence-gated."""

    TELEMETRY: ClassVar[dict] = {
        "interval_s": 1.0,
        "report_mb": 0.001,
        "energy_j_per_mb": 10.0,
        "max_loss_frac": 0.5,
        "evidence_mb_per_detection": 0.003,
    }

    def _run(self, tmp_path: Path, policy: str = "always_strong", **overrides):
        from tests.test_v2_visual_loop import scenario_payload, write_world_png

        world = write_world_png(tmp_path / "world.png")
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world, **overrides)))
        return MissionRunner(load_scenario(path), policy).run()

    def test_delivery_ledger_orders_visible_matched_delivered(self, tmp_path: Path) -> None:
        from aerointentbench.v2.mission_metrics import delivery_delays, timely_recall

        result = self._run(tmp_path, **{"simulation.telemetry": self.TELEMETRY})
        ledger = result.target_timeliness
        assert ledger["available"] is True
        delivered = {
            tid: t for tid, t in ledger["targets"].items() if t["first_delivered_s"] is not None
        }
        assert delivered, "fixture mission should deliver at least one target"
        for times in delivered.values():
            assert times["first_visible_s"] <= times["first_matched_s"]
            assert times["first_matched_s"] <= times["first_delivered_s"]
        delays = delivery_delays(result)
        assert all(d is None or d >= 0.0 for d in delays.values())
        # TR(T) is monotone in T.
        total = 1  # fixture has one target
        assert timely_recall(result, 0.0, total) <= timely_recall(result, 10.0, total)
        assert timely_recall(result, 10.0, total) <= timely_recall(result, 1e9, total)

    def test_outage_delays_or_denies_delivery(self, tmp_path: Path) -> None:
        dead = [
            {
                "regime_id": "disconnected",
                "start_s": 0.0,
                "uplink_mbps": 0.0,
                "downlink_mbps": 0.0,
                "rtt_ms": 0.0,
                "packet_loss_frac": 1.0,
            }
        ]
        result = self._run(
            tmp_path,
            **{"simulation.telemetry": self.TELEMETRY, "simulation.network_trace": dead},
        )
        # Matched, maybe; delivered, never — the link is down for the whole mission.
        assert all(
            t["first_delivered_s"] is None for t in result.target_timeliness["targets"].values()
        )

    def test_no_evidence_layer_is_a_loud_error(self, tmp_path: Path) -> None:
        from aerointentbench.v2.mission_metrics import delivery_delays

        result = self._run(tmp_path)  # no telemetry block at all
        assert result.target_timeliness["available"] is False
        with pytest.raises(ValueError, match="undefined"):
            delivery_delays(result)
