"""Split inference: head onboard, features across the link, tail on the server.

CI-safe coverage of the semantics that must hold regardless of the model: the
graph-cut payload is priced from real tensor bytes, head cost is charged on every
attempt (including failed uploads), feature reduction really reaches the tail,
fallback runs on the same frame, and — the reason split exists — a split config is
legal under ``features_only`` where raw-RGB offload is a privacy violation. The
heuristic partition here is a test fixture; the real torch partition is exercised by
the opt-in ``real_models`` test at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError  # noqa: E402
from aerointentbench.v2.runner import MissionRunner  # noqa: E402
from aerointentbench.v2.scenario import load_scenario  # noqa: E402
from aerointentbench.v2.split import encode_features  # noqa: E402
from tests.test_v2_visual_loop import scenario_payload, write_world_png  # noqa: E402

SPLIT_CONFIG = {
    "config_id": "SPLIT_STRONG",
    "model_strategy_id": "T_SPLIT_STRONG",
    "kind": "simulated_split",
    "mission_latency_s": 0.9,
    "energy_j_per_call": 3.0,
    "communication_mb_per_call": 0.0046,
    "quality_tier": "high",
    "parameters": {
        "backend_kind": "slow_strong",
        "backend_min_component_px": 6,
        "split_cut": "stage1",
        "head_latency_s": 0.2,
        "remote_compute_s": 0.1,
        "timeout_s": 3.0,
        "feature_dtype": "float16",
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


def _mission(tmp_path: Path, policy: str, *, extra_configs: list[dict], **overrides):
    world = write_world_png(tmp_path / "world.png")
    payload = scenario_payload(world, **overrides)
    payload["executor_configs"] = payload["executor_configs"] + extra_configs
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(payload))
    return MissionRunner(load_scenario(path), policy).run()


class TestSchema:
    def test_split_kind_loads_and_missing_parameters_fail(self, tmp_path: Path) -> None:
        result = _mission(tmp_path, "SPLIT_STRONG", extra_configs=[SPLIT_CONFIG])
        assert "SPLIT_STRONG" in result.config_selection_history
        broken = json.loads(json.dumps(SPLIT_CONFIG))
        del broken["parameters"]["split_cut"]
        with pytest.raises(SchemaValidationError, match="split_cut"):
            _mission(tmp_path, "SPLIT_STRONG", extra_configs=[broken])

    def test_split_backend_must_be_a_local_kind(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(SPLIT_CONFIG))
        broken["parameters"]["backend_kind"] = "simulated_remote"
        with pytest.raises(SchemaValidationError, match="backend_kind"):
            _mission(tmp_path, "SPLIT_STRONG", extra_configs=[broken])

    def test_unknown_feature_dtype_rejected(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(SPLIT_CONFIG))
        broken["parameters"]["feature_dtype"] = "int4"
        with pytest.raises(SchemaValidationError, match="feature_dtype"):
            _mission(tmp_path, "SPLIT_STRONG", extra_configs=[broken])


class TestPrivacy:
    """The reason split exists: features may leave the vehicle when raw input may not."""

    def test_features_only_permits_split_and_blocks_raw_remote(self, tmp_path: Path) -> None:
        split_run = _mission(
            tmp_path,
            "SPLIT_STRONG",
            extra_configs=[SPLIT_CONFIG, RAW_REMOTE_CONFIG],
            **{"mission_contract.privacy_level": "features_only"},
        )
        assert split_run.privacy_violation_count == 0
        assert split_run.constraints["privacy_constraint_success"] is True

        raw_run = _mission(
            tmp_path,
            "REMOTE_RAW",
            extra_configs=[SPLIT_CONFIG, RAW_REMOTE_CONFIG],
            **{"mission_contract.privacy_level": "features_only"},
        )
        assert raw_run.privacy_violation_count > 0
        assert raw_run.constraints["privacy_constraint_success"] is False

    def test_local_only_blocks_split_too(self, tmp_path: Path) -> None:
        result = _mission(
            tmp_path,
            "SPLIT_STRONG",
            extra_configs=[SPLIT_CONFIG],
            **{"mission_contract.privacy_level": "local_only"},
        )
        assert result.privacy_violation_count > 0


class TestPayloadAccounting:
    def test_payload_is_priced_from_real_tensor_bytes(self, tmp_path: Path) -> None:
        result = _mission(tmp_path, "SPLIT_STRONG", extra_configs=[SPLIT_CONFIG])
        log = result.observations[0]
        diag = log.execution["diagnostics"]
        assert diag["execution_location"] == "split"
        assert diag["split_cut"] == "stage1"
        # Heuristic head: 48x64 frame downsampled 2x -> (24, 32, 3) float16 features.
        expected_mb = 24 * 32 * 3 * 2 / 1e6
        assert diag["feature_payload_mb"] == pytest.approx(expected_mb)
        assert diag["request"]["payload_kind"] == "features"
        assert diag["request"]["payload_mb"] == pytest.approx(expected_mb)
        assert diag["uploaded_mb"] == pytest.approx(expected_mb)
        # Communication = features up + configured mask down.
        assert log.execution["communication_mb"] == pytest.approx(expected_mb + 0.01)

    def test_head_cost_is_charged_even_when_the_link_is_down(self, tmp_path: Path) -> None:
        config = json.loads(json.dumps(SPLIT_CONFIG))
        del config["parameters"]["download_mb_per_call"]
        result = _mission(
            tmp_path,
            "SPLIT_STRONG",
            extra_configs=[config],
            **{
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
            },
        )
        execution = result.observations[0].execution
        assert execution["diagnostics"]["remote_status"] == "unreachable"
        assert execution["success"] is False
        # The head ran before the probe: its energy and latency are spent regardless.
        assert execution["energy_j"] == pytest.approx(SPLIT_CONFIG["energy_j_per_call"])
        assert execution["mission_latency_s"] == pytest.approx(0.2 + 0.2)  # head + probe

    def test_fallback_runs_on_the_same_frame_after_a_dead_link(self, tmp_path: Path) -> None:
        config = json.loads(json.dumps(SPLIT_CONFIG))
        config["parameters"]["fallback_config_id"] = "FAST"
        result = _mission(
            tmp_path,
            "SPLIT_STRONG",
            extra_configs=[config],
            **{
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
            },
        )
        execution = result.observations[0].execution
        assert execution["success"] is True
        assert execution["diagnostics"]["fallback"]["fallback_config_id"] == "FAST"


class TestFeatureEncoding:
    def test_uint8_roundtrip_error_is_bounded_and_deterministic(self) -> None:
        rng = np.random.default_rng(5)
        tensors = {"cursor": rng.normal(0.0, 3.0, (7, 5, 4)).astype(np.float32)}
        first, mb, detail = encode_features(tensors, "uint8")
        second, _, _ = encode_features(tensors, "uint8")
        span = float(tensors["cursor"].max() - tensors["cursor"].min())
        assert np.array_equal(first["cursor"], second["cursor"])
        assert np.abs(first["cursor"] - tensors["cursor"]).max() <= span / 255.0
        assert mb == pytest.approx(7 * 5 * 4 / 1e6)
        assert detail["cursor"]["wire_bytes"] == 7 * 5 * 4

    def test_dtype_sizes_scale_as_expected(self) -> None:
        tensors = {"cursor": np.ones((10, 10), dtype=np.float32)}
        _, mb32, _ = encode_features(tensors, "float32")
        _, mb16, _ = encode_features(tensors, "float16")
        _, mb8, _ = encode_features(tensors, "uint8")
        assert mb32 == pytest.approx(2 * mb16) == pytest.approx(4 * mb8)


@pytest.mark.real_models
class TestTorchPartition:
    """The real cut: a float32 split must reproduce the full model's mask exactly."""

    def test_split_mask_matches_full_model(self) -> None:
        torch = pytest.importorskip("torch", reason="needs the [v2-real-models] extra")
        del torch
        from aerointentbench.v2.real_models import backend_for
        from aerointentbench.v2.scenario import ExecutorConfigSpec
        from aerointentbench.v2.split_models import TorchSplitPartition, legal_cuts

        backend_params = {
            "model_id": "lraspp_mobilenet_v3_large",
            "weights_id": "official_default",
            "task_class": "person",
            "device": "auto",
            "dtype": "float32",
            "input_width_px": 128,
            "input_height_px": 96,
            "warmup_runs": 0,
        }
        backend = backend_for(backend_params)
        cuts = legal_cuts(backend)
        assert len(cuts) >= 2

        spec = ExecutorConfigSpec(
            config_id="SPLIT_REAL",
            model_strategy_id="T_SPLIT_REAL",
            kind="simulated_split",
            mission_latency_s=0.5,
            energy_j_per_call=1.0,
            communication_mb_per_call=0.0,
            quality_tier="high",
            parameters={
                **{f"backend_{key}": value for key, value in backend_params.items()},
                "backend_probability_threshold": 0.5,
                "split_cut": cuts[len(cuts) // 2],
                "head_latency_s": 0.1,
                "remote_compute_s": 0.1,
                "timeout_s": 3.0,
            },
        )
        partition = TorchSplitPartition(spec)
        rng = np.random.default_rng(11)
        rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)

        probability_map, _ = backend.infer(rgb)
        full_mask = probability_map >= 0.5
        crossing = partition.run_head(rgb)
        split_mask = partition.run_tail(crossing, rgb.shape)
        assert split_mask.shape == full_mask.shape
        assert np.array_equal(split_mask, full_mask)
