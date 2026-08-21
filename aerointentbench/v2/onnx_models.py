"""Published ONNX segmentation models as onboard executors (incl. quantized releases).

Responsibility and boundaries: the ``onnx_semantic_segmentation`` executor kind runs a
published, locally cached ONNX artifact behind the same ``run(rgb)`` seam as the torch
kinds. The motivating catalog rows are the Qualcomm AI Hub releases of
DeepLabV3+-MobileNet (VOC2012, 21 classes), which ship **pre-quantized w8a8** next to
float — a PUBLISHED quantization, adopted like the pretrained splits are adopted:
provenance cited per config, never a quantization search performed here.

Rules mirrored from the torch kinds:

- ``onnxruntime`` lives in the ``[v2-onnx]`` extra and is imported lazily inside the
  backend; the default test suite injects a fake backend and never loads a model.
- The artifact is loaded UNMODIFIED from the local cache; sessions are opened at
  ``ORT_ENABLE_BASIC`` graph optimization because the QAIRT-exported QDQ graphs make
  ORT's full optimizer generate duplicate node names (an ORT/exporter interaction,
  recorded here rather than patched into the artifact).
- ``person_index`` comes from the model card and is a REQUIRED parameter — never
  hardcoded, exactly like the torch kinds resolve it from weight metadata.
- These artifacts emit an argmax class mask (uint8 ids), not probabilities, so there
  is no probability threshold; the person mask is ``classes == person_index``.
- ``latency_mode`` measured vs configured is explicit; energy stays simulated.
- Backends are cached per model path (no per-frame session rebuilds).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.executors import ImageExecutionResult
from aerointentbench.v2.real_models import (
    _drop_small_components,
    _resize_rgb_bilinear,
    resize_mask_nearest,
)
from aerointentbench.v2.scenario import ExecutorConfigSpec

__all__ = [
    "OnnxSemanticSegmentationExecutor",
    "onnx_backend_for",
    "validate_onnx_parameters",
]

_ONNX_REQUIRED_PARAMETERS = (
    "model_path",
    "person_index",
    "input_width_px",
    "input_height_px",
    "latency_mode",
)


def validate_onnx_parameters(parameters: Mapping[str, Any], context: str) -> None:
    missing = [key for key in _ONNX_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: an onnx_semantic_segmentation executor requires parameters {missing}"
        )
    if str(parameters["latency_mode"]) not in ("measured", "configured"):
        raise SchemaValidationError(f"{context}: latency_mode must be 'measured' or 'configured'")


class _OrtBackend:
    """One cached onnxruntime session over a published artifact."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort  # [v2-onnx] extra; lazy by design

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise SchemaValidationError(
                f"onnx model {path} not found. Published artifacts are cached locally "
                "and never committed; download the release named in the config's "
                "provenance (e.g. the Qualcomm AI Hub release zip) into that path."
            )
        options = ort.SessionOptions()
        # Full optimization trips 'two nodes with same node name' on QAIRT QDQ
        # exports; BASIC runs the artifact unmodified.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0]
        self._uint8_input = "uint8" in self._input.type
        self.info = {
            "model_path": str(path),
            "onnxruntime_input_type": self._input.type,
            "graph_optimization_level": "ORT_ENABLE_BASIC",
            "execution_provider": "CPUExecutionProvider",
        }

    def infer(self, rgb_resized: np.ndarray) -> tuple[np.ndarray, float]:
        """uint8 HWC RGB (model input size) -> (class-id mask HW, forward seconds).

        Published exports differ in output form: some emit an argmax class-id mask
        ``[1, H, W]`` (DeepLabV3+), others per-class logits ``[1, C, h, w]`` at reduced
        spatial resolution (SegFormer, FFNet). Logits are argmaxed here; the caller
        resizes the mask back to the observation, so the reduced grid is handled by
        the same nearest-neighbour path as every other backend.
        """
        chw = np.ascontiguousarray(rgb_resized.transpose(2, 0, 1))[np.newaxis]
        feed = chw if self._uint8_input else chw.astype(np.float32) / 255.0
        started = time.perf_counter()
        raw = self._session.run(None, {self._input.name: feed})[0]
        forward_s = time.perf_counter() - started
        raw = np.asarray(raw)[0]
        classes = raw.argmax(axis=0) if raw.ndim == 3 else raw
        return classes.astype(np.int32), forward_s


_BACKEND_CACHE: dict[str, _OrtBackend] = {}


def onnx_backend_for(parameters: Mapping[str, Any]) -> _OrtBackend:
    key = str(Path(str(parameters["model_path"])).expanduser())
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = _OrtBackend(key)
    return _BACKEND_CACHE[key]


class OnnxSemanticSegmentationExecutor:
    """The ``ImageExecutor`` over a published ONNX artifact. ``run(rgb)`` is the surface."""

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        *,
        backend: Any | None = None,
        backend_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.config_id = spec.config_id
        self.model_strategy_id = spec.model_strategy_id
        self._spec = spec
        params = spec.parameters
        self._input_w = int(params["input_width_px"])
        self._input_h = int(params["input_height_px"])
        self._person_index = int(params["person_index"])
        self._latency_mode = str(params["latency_mode"])
        self._min_component_px = int(params.get("min_component_px", 0))
        self._backend = (
            backend if backend is not None else (backend_factory or onnx_backend_for)(params)
        )

    def run(self, rgb: np.ndarray) -> ImageExecutionResult:
        started = time.perf_counter()
        observation_h, observation_w = rgb.shape[:2]
        resized = _resize_rgb_bilinear(rgb, self._input_w, self._input_h)
        classes, forward_s = self._backend.infer(resized)
        mask_input = classes == self._person_index
        if self._min_component_px > 0:
            mask_input = _drop_small_components(mask_input, self._min_component_px)
        mask = resize_mask_nearest(mask_input, observation_w, observation_h)
        end_to_end_s = time.perf_counter() - started
        measured_drives_clock = self._latency_mode == "measured"
        mission_latency_s = end_to_end_s if measured_drives_clock else self._spec.mission_latency_s
        return ImageExecutionResult(
            config_id=self.config_id,
            model_strategy_id=self.model_strategy_id,
            success=True,
            prediction_mask=mask,
            mission_latency_s=mission_latency_s,
            measured_wall_clock_s=end_to_end_s,
            energy_j=self._spec.energy_j_per_call,
            communication_mb=self._spec.communication_mb_per_call,
            measurement_provenance={
                "latency": (
                    "measured wall-clock drives the mission clock"
                    if measured_drives_clock
                    else "configured mission_latency_s drives the clock; wall-clock recorded"
                ),
                "energy": "configured (simulated)",
                "forward_s": round(forward_s, 4),
                "latency_mode": self._latency_mode,
                **self._backend.info,
            },
        )
