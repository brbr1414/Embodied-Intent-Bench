"""The onnx_semantic_segmentation kind: published (incl. quantized) ONNX artifacts.

CI-safe: a fake backend is injected (onnxruntime never imported); pinned are the
required-parameter validation, the argmax-class -> person-mask path (no probability
threshold exists on these artifacts), configured-vs-measured latency honesty, and
dispatch through build_executors.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.onnx_models import (
    OnnxSemanticSegmentationExecutor,
    validate_onnx_parameters,
)
from aerointentbench.v2.scenario import ExecutorConfigSpec

PARAMS = {
    "model_path": "~/.cache/qaihub/whatever.onnx",
    "person_index": 15,
    "input_width_px": 520,
    "input_height_px": 520,
    "latency_mode": "configured",
}


def _spec(**overrides) -> ExecutorConfigSpec:
    params = dict(PARAMS)
    params.update(overrides.pop("parameters", {}))
    values = {
        "config_id": "ONNX_TEST",
        "model_strategy_id": "QAIHUB_DLV3PLUS_W8A8",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.2,
        "energy_j_per_call": 3.0,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": params,
    }
    values.update(overrides)
    return ExecutorConfigSpec(**values)


class FakeOrtBackend:
    """Emits an argmax class-id mask with a person block, like the real artifacts."""

    info: ClassVar[dict] = {"model_path": "fake", "execution_provider": "fake"}

    def __init__(self, person_index: int = 15) -> None:
        self._person = person_index

    def infer(self, rgb_resized: np.ndarray):
        h, w = rgb_resized.shape[:2]
        classes = np.zeros((h, w), dtype=np.uint8)
        classes[h // 4 : h // 2, w // 4 : w // 2] = self._person
        return classes, 0.01


class TestValidation:
    def test_missing_parameter_is_rejected(self) -> None:
        bad = dict(PARAMS)
        del bad["person_index"]
        with pytest.raises(SchemaValidationError, match="person_index"):
            validate_onnx_parameters(bad, "ctx")

    def test_bad_latency_mode_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="latency_mode"):
            validate_onnx_parameters(dict(PARAMS, latency_mode="guessed"), "ctx")


class TestExecution:
    def test_person_mask_comes_from_class_ids(self) -> None:
        executor = OnnxSemanticSegmentationExecutor(_spec(), backend=FakeOrtBackend())
        rgb = np.zeros((192, 256, 3), dtype=np.uint8)
        result = executor.run(rgb)
        assert result.success
        assert result.prediction_mask.shape == (192, 256)
        assert result.prediction_mask.any()
        assert result.mission_latency_s == 0.2  # configured drives the clock
        assert result.measurement_provenance["latency_mode"] == "configured"

    def test_other_person_index_yields_empty_mask(self) -> None:
        executor = OnnxSemanticSegmentationExecutor(
            _spec(parameters={"person_index": 7}), backend=FakeOrtBackend(person_index=15)
        )
        result = executor.run(np.zeros((192, 256, 3), dtype=np.uint8))
        assert not result.prediction_mask.any()

    def test_measured_mode_uses_wall_clock(self) -> None:
        executor = OnnxSemanticSegmentationExecutor(
            _spec(parameters={"latency_mode": "measured"}), backend=FakeOrtBackend()
        )
        result = executor.run(np.zeros((192, 256, 3), dtype=np.uint8))
        assert result.mission_latency_s == result.measured_wall_clock_s


class TestLogitsOutput:
    def test_logit_emitting_artifacts_are_argmaxed(self) -> None:
        class LogitsBackend:
            info: ClassVar[dict] = {"model_path": "fake-logits"}

            def infer(self, rgb_resized):
                h, w = rgb_resized.shape[:2]
                logits = np.zeros((21, h // 4, w // 4), dtype=np.float32)
                logits[15, 2:6, 2:6] = 5.0  # person wins in a block
                return logits.argmax(axis=0), 0.01

        executor = OnnxSemanticSegmentationExecutor(_spec(), backend=LogitsBackend())
        result = executor.run(np.zeros((192, 256, 3), dtype=np.uint8))
        assert result.prediction_mask.shape == (192, 256)
        assert result.prediction_mask.any()


class TestDispatch:
    def test_build_executors_routes_the_kind(self) -> None:
        from aerointentbench.v2 import onnx_models
        from aerointentbench.v2.executors import build_executors

        spec = _spec()
        original = onnx_models.onnx_backend_for
        onnx_models.onnx_backend_for = lambda params: FakeOrtBackend()
        try:
            registry = build_executors((spec,))
        finally:
            onnx_models.onnx_backend_for = original
        assert "ONNX_TEST" in registry
