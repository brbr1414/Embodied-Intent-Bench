"""Pre-split models: literature-defined split configurations as catalog options.

CI-safe coverage of the semantics that must hold regardless of the model: split
provenance is part of the schema (a predefined split without a citable source is
rejected at load), the wire payload is priced from the bytes the head actually
encoded, head cost is charged on every attempt (including failed uploads),
fallback runs on the same frame, the result honours the shared
``ImageExecutionResult`` contract, and a pretrained-split config is legal under
``features_only`` where raw-RGB offload is a violation. The fixture backend here
is a test fixture; the genuine sc2 checkpoint path is exercised by the opt-in
``real_models`` test at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError  # noqa: E402
from aerointentbench.v2.presplit import SplitSpec  # noqa: E402
from aerointentbench.v2.runner import MissionRunner  # noqa: E402
from aerointentbench.v2.scenario import load_scenario  # noqa: E402
from tests.test_v2_visual_loop import scenario_payload, write_world_png  # noqa: E402

PRESPLIT_CONFIG = {
    "config_id": "PRESPLIT_ES",
    "model_strategy_id": "T_PRESPLIT_ES",
    "kind": "pretrained_split",
    "mission_latency_s": 0.9,
    "energy_j_per_call": 3.0,
    "communication_mb_per_call": 0.0002,
    "quality_tier": "high",
    "parameters": {
        "split_backend": "fixture",
        "backend_kind": "slow_strong",
        "split_source": "paper",
        "split_source_reference": "Matsubara et al., SC2 Benchmark, TMLR 2023",
        "split_location": "bottleneck replaces conv1..layer1 of ResNet-50",
        "head_latency_s": 0.2,
        "remote_compute_s": 0.1,
        "timeout_s": 3.0,
        "download_mb_per_call": 0.01,
    },
}

RAW_REMOTE_CONFIG = {
    "config_id": "REMOTE_RAW",
    "model_strategy_id": "T_REMOTE_RAW",
    "kind": "simulated_remote",
    "mission_latency_s": 0.9,
    "energy_j_per_call": 0.0,
    "communication_mb_per_call": 2.0,
    "quality_tier": "high",
    "parameters": {
        "backend_kind": "slow_strong",
        "remote_compute_s": 0.1,
        "timeout_s": 3.0,
    },
}

DEAD_TRACE = {
    "simulation.network_trace": [
        {
            "regime_id": "disconnected",
            "start_s": 0.0,
            "uplink_mbps": 0.0,
            "downlink_mbps": 0.0,
            "rtt_ms": 0.0,
            "packet_loss_frac": 1.0,
        }
    ]
}


def _mission(tmp_path: Path, policy: str, *, extra_configs: list[dict], **overrides):
    world = write_world_png(tmp_path / "world.png")
    payload = scenario_payload(world, **overrides)
    payload["executor_configs"] = payload["executor_configs"] + extra_configs
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(payload))
    return MissionRunner(load_scenario(path), policy).run()


class TestSchema:
    def test_presplit_kind_loads_and_runs(self, tmp_path: Path) -> None:
        result = _mission(tmp_path, "PRESPLIT_ES", extra_configs=[PRESPLIT_CONFIG])
        assert "PRESPLIT_ES" in result.config_selection_history

    @pytest.mark.parametrize(
        "missing",
        ["split_source", "split_source_reference", "split_location", "split_backend"],
    )
    def test_missing_provenance_fails_at_load(self, tmp_path: Path, missing: str) -> None:
        broken = json.loads(json.dumps(PRESPLIT_CONFIG))
        del broken["parameters"][missing]
        with pytest.raises(SchemaValidationError, match="pretrained_split"):
            _mission(tmp_path, "PRESPLIT_ES", extra_configs=[broken])

    def test_benchmark_discovered_split_source_is_rejected(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(PRESPLIT_CONFIG))
        broken["parameters"]["split_source"] = "optimized_by_benchmark"
        with pytest.raises(SchemaValidationError, match="never discovered"):
            _mission(tmp_path, "PRESPLIT_ES", extra_configs=[broken])

    def test_blank_reference_is_rejected(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(PRESPLIT_CONFIG))
        broken["parameters"]["split_source_reference"] = "   "
        with pytest.raises(SchemaValidationError, match="split_source_reference"):
            _mission(tmp_path, "PRESPLIT_ES", extra_configs=[broken])

    def test_unknown_split_backend_is_rejected(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(PRESPLIT_CONFIG))
        broken["parameters"]["split_backend"] = "mystery"
        with pytest.raises(SchemaValidationError, match="split_backend"):
            _mission(tmp_path, "PRESPLIT_ES", extra_configs=[broken])

    def test_split_spec_roundtrip(self) -> None:
        spec = SplitSpec.from_parameters("CFG", dict(PRESPLIT_CONFIG["parameters"]))
        assert spec.to_dict() == {
            "split_id": "CFG",
            "split_location": "bottleneck replaces conv1..layer1 of ResNet-50",
            "source": "paper",
            "source_reference": "Matsubara et al., SC2 Benchmark, TMLR 2023",
        }


class TestPrivacy:
    """A predefined split transmits features: legal under features_only, not local_only."""

    def test_features_only_permits_presplit_and_blocks_raw_remote(self, tmp_path: Path) -> None:
        presplit_run = _mission(
            tmp_path,
            "PRESPLIT_ES",
            extra_configs=[PRESPLIT_CONFIG, RAW_REMOTE_CONFIG],
            **{"mission_contract.privacy_level": "features_only"},
        )
        assert presplit_run.privacy_violation_count == 0
        assert presplit_run.constraints["privacy_constraint_success"] is True

        raw_run = _mission(
            tmp_path,
            "REMOTE_RAW",
            extra_configs=[PRESPLIT_CONFIG, RAW_REMOTE_CONFIG],
            **{"mission_contract.privacy_level": "features_only"},
        )
        assert raw_run.privacy_violation_count > 0

    def test_local_only_blocks_presplit(self, tmp_path: Path) -> None:
        result = _mission(
            tmp_path,
            "PRESPLIT_ES",
            extra_configs=[PRESPLIT_CONFIG],
            **{"mission_contract.privacy_level": "local_only"},
        )
        assert result.privacy_violation_count > 0


class TestAccounting:
    def test_payload_is_priced_from_real_encoded_bytes(self, tmp_path: Path) -> None:
        result = _mission(tmp_path, "PRESPLIT_ES", extra_configs=[PRESPLIT_CONFIG])
        log = result.observations[0]
        diag = log.execution["diagnostics"]
        assert diag["execution_location"] == "split"
        assert diag["split_spec"]["source"] == "paper"
        assert diag["fixture"] is True
        # Fixture head: 48x64 frame -> 4x-downsampled grayscale -> 12x16 uint8 bytes.
        expected_mb = 12 * 16 / 1e6
        assert diag["payload_bytes"] == 12 * 16
        assert diag["feature_payload_mb"] == pytest.approx(expected_mb)
        assert diag["request"]["payload_kind"] == "features"
        assert diag["uploaded_mb"] == pytest.approx(expected_mb)
        # Communication = encoded features up + configured mask down; NOT the nominal
        # communication_mb_per_call.
        assert log.execution["communication_mb"] == pytest.approx(expected_mb + 0.01)

    def test_head_cost_is_charged_even_when_the_link_is_down(self, tmp_path: Path) -> None:
        config = json.loads(json.dumps(PRESPLIT_CONFIG))
        del config["parameters"]["download_mb_per_call"]
        result = _mission(tmp_path, "PRESPLIT_ES", extra_configs=[config], **DEAD_TRACE)
        execution = result.observations[0].execution
        assert execution["diagnostics"]["remote_status"] == "unreachable"
        assert execution["success"] is False
        assert execution["energy_j"] == pytest.approx(PRESPLIT_CONFIG["energy_j_per_call"])
        assert execution["mission_latency_s"] == pytest.approx(0.2 + 0.2)  # head + probe

    def test_fallback_runs_on_the_same_frame_after_a_dead_link(self, tmp_path: Path) -> None:
        config = json.loads(json.dumps(PRESPLIT_CONFIG))
        config["parameters"]["fallback_config_id"] = "FAST"
        result = _mission(tmp_path, "PRESPLIT_ES", extra_configs=[config], **DEAD_TRACE)
        execution = result.observations[0].execution
        assert execution["success"] is True
        assert execution["diagnostics"]["fallback"]["fallback_config_id"] == "FAST"


class TestContractAndDeterminism:
    def test_result_shape_matches_other_executors(self, tmp_path: Path) -> None:
        presplit = _mission(tmp_path, "PRESPLIT_ES", extra_configs=[PRESPLIT_CONFIG])
        onboard = _mission(tmp_path, "FAST", extra_configs=[PRESPLIT_CONFIG])
        base_keys = set(onboard.observations[0].execution) - {"diagnostics"}
        assert base_keys <= set(presplit.observations[0].execution)

    def test_missions_are_deterministic(self, tmp_path: Path) -> None:
        def strip_wall_clock(value):
            if isinstance(value, dict):
                return {
                    k: strip_wall_clock(v)
                    for k, v in value.items()
                    if k != "measured_wall_clock_s"  # the one declared-measured diagnostic
                }
            if isinstance(value, list):
                return [strip_wall_clock(v) for v in value]
            return value

        first = _mission(tmp_path / "a", "PRESPLIT_ES", extra_configs=[PRESPLIT_CONFIG])
        second = _mission(tmp_path / "b", "PRESPLIT_ES", extra_configs=[PRESPLIT_CONFIG])
        assert json.dumps(strip_wall_clock(first.to_dict()), sort_keys=True) == json.dumps(
            strip_wall_clock(second.to_dict()), sort_keys=True
        )

    def test_v1_policy_runs_over_a_catalog_containing_presplit(self, tmp_path: Path) -> None:
        result = _mission(
            tmp_path,
            "rule_based",
            extra_configs=[PRESPLIT_CONFIG],
            **{"mission_contract.privacy_level": "features_only"},
        )
        assert len(result.config_selection_history) > 0
        # The catalog exposed both onboard and split modes to the policy; whichever it
        # picked, every selection resolved to a runnable executor.
        assert set(result.config_selection_history) <= {"FAST", "STRONG", "PRESPLIT_ES"}


@pytest.mark.real_models
class TestGenuineSc2Backend:
    """The genuine checkpoint path: encode -> entropy-coded bytes -> decode -> mask."""

    def test_entropic_student_checkpoint_roundtrip(self) -> None:
        pytest.importorskip("torch", reason="needs the [v2-presplit] extra")
        pytest.importorskip("sc2bench", reason="needs the [v2-presplit] extra")
        from aerointentbench.v2.presplit import _Sc2EntropicStudentModel
        from aerointentbench.v2.scenario import ExecutorConfigSpec

        checkpoint = (
            Path.home()
            / ".cache/sc2-benchmark/resource/ckpt/pascal_voc2012/supervised_compression"
            / "entropic_student"
            / "pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta5.12_from_deeplabv3_resnet50.pt"
        )
        if not checkpoint.exists():
            pytest.skip("sc2-benchmark checkpoint not present locally")

        spec = ExecutorConfigSpec(
            config_id="PRESPLIT_REAL",
            model_strategy_id="ES_DLV3_R50",
            kind="pretrained_split",
            mission_latency_s=0.5,
            energy_j_per_call=1.0,
            communication_mb_per_call=0.001,
            quality_tier="low",
            parameters={
                "split_backend": "sc2_entropic_student",
                "split_source": "paper",
                "split_source_reference": "Matsubara et al., SC2 Benchmark, TMLR 2023",
                "split_location": "bottleneck replaces conv1..layer1 of ResNet-50",
                "checkpoint_path": str(checkpoint),
                "input_width_px": 128,
                "input_height_px": 96,
                "device": "cpu",
                "head_latency_s": 0.05,
                "remote_compute_s": 0.1,
                "timeout_s": 3.0,
            },
        )
        model = _Sc2EntropicStudentModel(spec)
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        payload, payload_mb, diagnostics = model.encode(rgb)
        assert diagnostics["payload_bytes"] > 0
        assert payload_mb == pytest.approx(diagnostics["payload_bytes"] / 1e6)
        mask = model.complete(payload, rgb.shape)
        assert mask.shape == (96, 128)
        assert mask.dtype == np.bool_
        # Same frame encodes to the same bitstream: deterministic.
        _, second_mb, _ = model.encode(rgb)
        assert second_mb == payload_mb
