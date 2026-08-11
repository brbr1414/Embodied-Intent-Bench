"""Instance segmentation (Mask R-CNN) as a catalog family: onboard kind + FCM split.

CI-safe coverage uses an injected fake backend (no torch): schema validation of
the new local kind, the shared result contract, and mission integration. The
genuine weights path — including the exactness pin that a float32 FCM split
reproduces the onboard model's mask — is the opt-in ``real_models`` test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError  # noqa: E402
from aerointentbench.v2.executors import ImageExecutionResult  # noqa: E402
from aerointentbench.v2.instance_models import TorchInstanceSegmentationExecutor  # noqa: E402
from aerointentbench.v2.runner import MissionRunner  # noqa: E402
from aerointentbench.v2.scenario import ExecutorConfigSpec, load_scenario  # noqa: E402
from tests.test_v2_visual_loop import scenario_payload, write_world_png  # noqa: E402

INSTANCE_CONFIG = {
    "config_id": "MASKRCNN_ONBOARD",
    "model_strategy_id": "T_MASKRCNN",
    "kind": "torch_instance_segmentation",
    "mission_latency_s": 1.2,
    "energy_j_per_call": 20.0,
    "communication_mb_per_call": 0.0,
    "quality_tier": "high",
    "parameters": {
        "score_threshold": 0.5,
        "input_min_size_px": 96,
        "input_max_size_px": 128,
        "device": "cpu",
        "latency_mode": "configured",
    },
}


class _FakeInstanceBackend:
    """Deterministic stand-in: 'detects' a fixed square when the threshold allows."""

    def person_mask(self, rgb: np.ndarray, score_threshold: float):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if score_threshold <= 0.9:
            mask[2:10, 3:12] = True
        return mask, 0.001


def _spec(**overrides) -> ExecutorConfigSpec:
    payload = json.loads(json.dumps(INSTANCE_CONFIG))
    payload["parameters"].update(overrides)
    return ExecutorConfigSpec(
        config_id=payload["config_id"],
        model_strategy_id=payload["model_strategy_id"],
        kind=payload["kind"],
        mission_latency_s=payload["mission_latency_s"],
        energy_j_per_call=payload["energy_j_per_call"],
        communication_mb_per_call=payload["communication_mb_per_call"],
        quality_tier=payload["quality_tier"],
        parameters=payload["parameters"],
    )


class TestSchema:
    @pytest.mark.parametrize(
        "missing",
        ["score_threshold", "input_min_size_px", "input_max_size_px", "device", "latency_mode"],
    )
    def test_missing_parameters_fail_at_load(self, tmp_path: Path, missing: str) -> None:
        world = write_world_png(tmp_path / "world.png")
        payload = scenario_payload(world)
        broken = json.loads(json.dumps(INSTANCE_CONFIG))
        del broken["parameters"][missing]
        payload["executor_configs"] = payload["executor_configs"] + [broken]
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(SchemaValidationError, match="torch_instance_segmentation"):
            load_scenario(path)

    def test_bad_latency_mode_and_threshold_fail(self, tmp_path: Path) -> None:
        world = write_world_png(tmp_path / "world.png")
        for override, match in (
            ({"latency_mode": "guessed"}, "latency_mode"),
            ({"score_threshold": 1.5}, "score_threshold"),
        ):
            payload = scenario_payload(world)
            broken = json.loads(json.dumps(INSTANCE_CONFIG))
            broken["parameters"].update(override)
            payload["executor_configs"] = payload["executor_configs"] + [broken]
            path = tmp_path / "scenario.json"
            path.write_text(json.dumps(payload))
            with pytest.raises(SchemaValidationError, match=match):
                load_scenario(path)


class TestExecutorContract:
    def test_result_follows_the_shared_contract(self) -> None:
        executor = TorchInstanceSegmentationExecutor(_spec(), backend=_FakeInstanceBackend())
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        result = executor.run(rgb)
        assert isinstance(result, ImageExecutionResult)
        assert result.success is True
        assert result.prediction_mask.shape == (48, 64)
        assert result.prediction_mask.dtype == np.bool_
        assert result.mission_latency_s == INSTANCE_CONFIG["mission_latency_s"]  # configured
        assert result.energy_j == INSTANCE_CONFIG["energy_j_per_call"]
        assert result.diagnostics["executor_kind"] == "torch_instance_segmentation"
        assert result.diagnostics["predicted_positive_px"] == 8 * 9

    def test_threshold_reaches_the_backend(self) -> None:
        executor = TorchInstanceSegmentationExecutor(
            _spec(score_threshold=0.95), backend=_FakeInstanceBackend()
        )
        result = executor.run(np.zeros((48, 64, 3), dtype=np.uint8))
        assert result.diagnostics["predicted_positive_px"] == 0


class TestMissionIntegration:
    def test_mission_runs_with_injected_fake_backend(self, tmp_path: Path, monkeypatch) -> None:
        import aerointentbench.v2.instance_models as instance_models

        monkeypatch.setattr(
            instance_models, "instance_backend_for", lambda params: _FakeInstanceBackend()
        )
        world = write_world_png(tmp_path / "world.png")
        payload = scenario_payload(world)
        payload["executor_configs"] = payload["executor_configs"] + [INSTANCE_CONFIG]
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(payload))
        result = MissionRunner(load_scenario(path), "MASKRCNN_ONBOARD").run()
        assert "MASKRCNN_ONBOARD" in result.config_selection_history


@pytest.mark.real_models
class TestGenuineMaskRcnn:
    """Real weights: the float32 FCM split must reproduce the onboard mask exactly."""

    def test_fcm_split_fp32_matches_onboard_full(self) -> None:
        pytest.importorskip("torch", reason="needs the [v2-real-models] extra")
        pytest.importorskip("torchvision", reason="needs the [v2-real-models] extra")
        from aerointentbench.v2.instance_models import FcmMaskRcnnSplitModel

        onboard = TorchInstanceSegmentationExecutor(_spec())
        rng = np.random.default_rng(11)
        rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        full_result = onboard.run(rgb)

        split_spec = ExecutorConfigSpec(
            config_id="MASKRCNN_SPLIT",
            model_strategy_id="T_MASKRCNN_SPLIT",
            kind="pretrained_split",
            mission_latency_s=0.8,
            energy_j_per_call=10.0,
            communication_mb_per_call=1.0,
            quality_tier="high",
            parameters={
                "split_backend": "fcm_maskrcnn_fpn",
                "split_source": "paper",
                "split_source_reference": "MPEG FCM split point via CompressAI-Vision",
                "split_location": "FPN outputs",
                "score_threshold": 0.5,
                "input_min_size_px": 96,
                "input_max_size_px": 128,
                "device": "cpu",
                "feature_dtype": "float32",
                "head_latency_s": 0.2,
                "remote_compute_s": 0.3,
                "timeout_s": 3.0,
            },
        )
        model = FcmMaskRcnnSplitModel(split_spec)
        payload, payload_mb, diagnostics = model.encode(rgb)
        assert payload_mb > 0
        assert len(diagnostics["feature_payload"]) >= 4  # multi-scale P-layers
        split_mask = model.complete(payload, rgb.shape)
        assert np.array_equal(split_mask, full_result.prediction_mask)


@pytest.mark.real_models
class TestGenuineGhndBq:
    """Real GHND-BQ checkpoint: encode -> quantized latent bytes -> decode -> mask."""

    def test_ghnd_bq_checkpoint_roundtrip(self) -> None:
        pytest.importorskip("torch", reason="needs the [v2-presplit] extra")
        pytest.importorskip("sc2bench", reason="needs the [v2-presplit] extra")
        from aerointentbench.v2.presplit import _Sc2GhndBqModel

        checkpoint = (
            Path.home()
            / ".cache/sc2-benchmark/resource/ckpt/pascal_voc2012/supervised_compression"
            / "ghnd-bq/pascal_voc2012-deeplabv3_resnet50-bq3ch_from_deeplabv3_resnet50.pt"
        )
        if not checkpoint.exists():
            pytest.skip("sc2-benchmark GHND-BQ checkpoint not present locally")

        spec = ExecutorConfigSpec(
            config_id="GHND",
            model_strategy_id="GHND_BQ3",
            kind="pretrained_split",
            mission_latency_s=0.4,
            energy_j_per_call=8.0,
            communication_mb_per_call=0.04,
            quality_tier="medium",
            parameters={
                "split_backend": "sc2_ghnd_bq",
                "split_source": "paper",
                "split_source_reference": "Matsubara et al., IEEE Access 2020 (GHND)",
                "split_location": "GHND bottleneck replaces conv1..layer1",
                "checkpoint_path": str(checkpoint),
                "bottleneck_channels": 3,
                "input_width_px": 128,
                "input_height_px": 96,
                "device": "cpu",
                "head_latency_s": 0.05,
                "remote_compute_s": 0.2,
                "timeout_s": 3.0,
            },
        )
        model = _Sc2GhndBqModel(spec)
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        payload, _payload_mb, diagnostics = model.encode(rgb)
        assert diagnostics["payload_bytes"] > 0
        # 8-bit latent: exactly one byte per element, no entropy coding.
        latent = payload["z"]
        assert diagnostics["payload_bytes"] == int(getattr(latent, "tensor", latent).numel())
        mask = model.complete(payload, rgb.shape)
        assert mask.shape == (96, 128)
        assert mask.dtype == np.bool_
