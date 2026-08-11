"""Instance segmentation (Mask R-CNN) as a catalog family: onboard and FCM split.

Two deployment options for torchvision's Mask R-CNN R50-FPN, both consuming the
official DEFAULT weights (no training, no tuning):

- ``torch_instance_segmentation`` — the local executor kind: the whole model runs
  onboard and the person instances above the score threshold are merged into the
  benchmark's boolean person mask (the person class index comes from the weight
  metadata, never hardcoded).
- ``FcmMaskRcnnSplitModel`` — the ``pretrained_split`` backend: backbone + FPN run
  onboard, the multi-scale P-layer feature tensors cross the link, and the RPN +
  ROI heads run on the server. The split point is NOT ours: it is the split-
  inference point standardized by the MPEG FCM (Feature Coding for Machines)
  activity and implemented by CompressAI-Vision for R-CNN models. What IS ours —
  and is honestly labelled in the SplitSpec — is the feature reduction: the
  standard's learned/codec feature compression (FCTM/VVC) is out of scope, so the
  crossing tensors use the same per-tensor affine quantization as the graph-cut
  split (``aerointentbench.v2.split.encode_features``), priced from real bytes.
  The resulting payload is deliberately large compared to a trained bottleneck —
  that gap (standard split point without a learned codec vs a paper's supervised
  compression) is a benchmark finding, not a defect.

Heavy imports stay lazy; torch/torchvision live in the ``[v2-real-models]``
extra. The default suite uses injected fakes and downloads nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol

import numpy as np

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.executors import ImageExecutionResult
from aerointentbench.v2.scenario import ExecutorConfigSpec

__all__ = [
    "FcmMaskRcnnSplitModel",
    "InstanceSegmentationBackend",
    "TorchInstanceSegmentationExecutor",
    "instance_backend_for",
]

_INSTANCE_REQUIRED_PARAMETERS: Final = (
    "score_threshold",
    "input_min_size_px",
    "input_max_size_px",
    "device",
    "latency_mode",
)

#: Process-wide cache: one loaded Mask R-CNN per (device, min, max), shared by the
#: onboard executor and the split backend so the checkpoint loads at most once.
_INSTANCE_BACKEND_CACHE: dict[tuple[str, int, int], Any] = {}


class InstanceSegmentationBackend(Protocol):
    """Person-instance inference over an RGB array: mask union + forward seconds."""

    def person_mask(self, rgb: np.ndarray, score_threshold: float) -> tuple[np.ndarray, float]: ...


class _TorchMaskRcnnBackend:
    """torchvision ``maskrcnn_resnet50_fpn`` with official DEFAULT weights."""

    def __init__(self, device: str, min_size: int, max_size: int) -> None:
        try:
            import torch
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_Weights,
                maskrcnn_resnet50_fpn,
            )
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise SchemaValidationError(
                "a torch_instance_segmentation executor needs the [v2-real-models] extra "
                "(torch + torchvision)"
            ) from error

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.person_index = weights.meta["categories"].index("person")
        self.weights_id = str(weights)
        resolved = device
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved
        self._model = maskrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
        self._model.to(resolved).eval()

    @property
    def model(self) -> Any:
        return self._model

    def _to_tensor(self, rgb: np.ndarray) -> Any:
        import torch

        array = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1)).to(self.device)

    def person_mask(self, rgb: np.ndarray, score_threshold: float) -> tuple[np.ndarray, float]:
        import torch

        tensor = self._to_tensor(rgb)
        started = time.perf_counter()
        with torch.no_grad():
            detections = self._model([tensor])[0]
        forward_s = time.perf_counter() - started
        return self._merge_person_masks(detections, rgb.shape, score_threshold), forward_s

    def _merge_person_masks(
        self, detections: Mapping[str, Any], rgb_shape: tuple[int, ...], score_threshold: float
    ) -> np.ndarray:
        mask = np.zeros(rgb_shape[:2], dtype=bool)
        labels = detections["labels"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        for index in np.flatnonzero((labels == self.person_index) & (scores >= score_threshold)):
            instance = detections["masks"][int(index), 0].cpu().numpy() >= 0.5
            mask |= instance
        return mask


def instance_backend_for(parameters: Mapping[str, Any]) -> _TorchMaskRcnnBackend:
    """Return the cached Mask R-CNN backend, loading weights at most once."""
    device = str(parameters["device"])
    min_size = int(parameters["input_min_size_px"])
    max_size = int(parameters["input_max_size_px"])
    key = (device, min_size, max_size)
    if key not in _INSTANCE_BACKEND_CACHE:
        _INSTANCE_BACKEND_CACHE[key] = _TorchMaskRcnnBackend(device, min_size, max_size)
    return _INSTANCE_BACKEND_CACHE[key]


class TorchInstanceSegmentationExecutor:
    """The ``ImageExecutor`` over a real instance-segmentation backend.

    Same surface and provenance rules as the semantic torch executor: ``run(rgb)``
    is everything, mission latency is configured unless ``latency_mode`` is
    ``measured``, and energy stays configured — elapsed time is not a power meter.
    """

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        *,
        backend: InstanceSegmentationBackend | None = None,
        backend_factory: Callable[[Mapping[str, Any]], InstanceSegmentationBackend] | None = None,
    ) -> None:
        self.config_id = spec.config_id
        self.model_strategy_id = spec.model_strategy_id
        self._spec = spec
        params = spec.parameters
        self._score_threshold = float(params["score_threshold"])
        self._latency_mode = str(params["latency_mode"])
        #: Injectable for tests; the real path loads (or reuses) the cached backend.
        if backend is not None:
            self._backend = backend
        else:
            self._backend = (backend_factory or instance_backend_for)(params)

    def run(self, rgb: np.ndarray) -> ImageExecutionResult:
        started = time.perf_counter()
        mask, forward_s = self._backend.person_mask(rgb, self._score_threshold)
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
                "mission_latency_s": (
                    "measured (end-to-end executor latency drives the mission clock; "
                    "machine-dependent)"
                    if measured_drives_clock
                    else "simulated (configured per executor; measured latency recorded "
                    "as diagnostic)"
                ),
                "measured_wall_clock_s": "measured (end-to-end: transfer + forward + merge)",
                "energy_j": "simulated (configured per call; elapsed time is not a power meter)",
                "communication_mb": "configured (local executor; no real transmission)",
            },
            diagnostics={
                "executor_kind": "torch_instance_segmentation",
                "score_threshold": self._score_threshold,
                "latency_mode": self._latency_mode,
                "model_forward_latency_s": round(forward_s, 6),
                "predicted_positive_px": int(mask.sum()),
            },
        )


class FcmMaskRcnnSplitModel:
    """Mask R-CNN split at the FPN outputs — the MPEG FCM standard split point.

    ``encode``: torchvision's detection transform + backbone + FPN run "onboard";
    every P-layer tensor is reduced with the shared per-tensor affine
    quantization and priced from real bytes. ``complete``: the RPN + ROI heads
    consume the (round-tripped) features "server-side" and the person instances
    above the score threshold merge into the boolean mask at capture resolution.
    Image-size bookkeeping (a handful of integers) travels as unpriced metadata,
    like the quantization scales.
    """

    def __init__(self, spec: ExecutorConfigSpec) -> None:
        params = dict(spec.parameters)
        missing = [
            key
            for key in ("score_threshold", "input_min_size_px", "input_max_size_px", "device")
            if key not in params
        ]
        if missing:
            raise SchemaValidationError(
                f"config {spec.config_id!r}: fcm_maskrcnn_fpn requires parameters {missing}"
            )
        self._spec = spec
        self._score_threshold = float(params["score_threshold"])
        self._feature_dtype = str(params.get("feature_dtype", "uint8"))
        self._backend = instance_backend_for(params)
        self.model_strategy_id = spec.model_strategy_id

    def encode(self, rgb: np.ndarray) -> tuple[Any, float, dict[str, Any]]:
        import torch

        from aerointentbench.v2.split import encode_features

        model = self._backend.model
        tensor = self._backend._to_tensor(rgb)
        with torch.no_grad():
            images, _ = model.transform([tensor], None)
            features = model.backbone(images.tensors)
        crossing = {
            name: feature[0].cpu().numpy() for name, feature in features.items()
        }  # one image per call
        dequantized, payload_mb, detail = encode_features(crossing, self._feature_dtype)
        payload = {
            "features": dequantized,
            "padded_shape": tuple(int(v) for v in images.tensors.shape),
            "image_size": tuple(int(v) for v in images.image_sizes[0]),
            "original_size": (int(rgb.shape[0]), int(rgb.shape[1])),
        }
        return (
            payload,
            payload_mb,
            {
                "payload_bytes": int(payload_mb * 1e6),
                "feature_dtype": self._feature_dtype,
                "feature_payload": detail,
            },
        )

    def complete(self, payload: Any, rgb_shape: tuple[int, ...]) -> np.ndarray:
        import torch
        from torchvision.models.detection.image_list import ImageList

        model = self._backend.model
        device = self._backend.device
        features = {
            name: torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)
            for name, array in payload["features"].items()
        }
        # The server needs the padded canvas shape for anchor placement; a zero
        # tensor of that shape carries it without re-transmitting the image.
        canvas = torch.zeros(payload["padded_shape"], device=device)
        image_list = ImageList(canvas, [payload["image_size"]])
        with torch.no_grad():
            proposals, _ = model.rpn(image_list, features, None)
            detections, _ = model.roi_heads(features, proposals, [payload["image_size"]], None)
            detections = model.transform.postprocess(
                detections, [payload["image_size"]], [payload["original_size"]]
            )[0]
        return self._backend._merge_person_masks(detections, rgb_shape, self._score_threshold)


def validate_instance_parameters(parameters: Mapping[str, Any], context: str) -> None:
    """Validate a torch_instance_segmentation config's parameters at load time."""
    missing = [key for key in _INSTANCE_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: a torch_instance_segmentation executor requires parameters "
            f"{list(_INSTANCE_REQUIRED_PARAMETERS)}; missing {missing}"
        )
    if parameters["latency_mode"] not in ("measured", "configured"):
        raise SchemaValidationError(
            f"{context}: latency_mode {parameters['latency_mode']!r} must be 'measured' or "
            "'configured'"
        )
    threshold = parameters["score_threshold"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not (0.0 <= threshold <= 1.0)
    ):
        raise SchemaValidationError(
            f"{context}: score_threshold must be a number in [0, 1], got {threshold!r}"
        )
