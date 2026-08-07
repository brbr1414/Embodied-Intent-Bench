"""Real split partitions of the torchvision segmentation models.

Cuts a cached :class:`TorchSegmentationBackend` model into an onboard head and a
server tail at a **legal cut**: a boundary between the backbone's ordered children
(torchvision's ``IntermediateLayerGetter`` preserves definition order). The crossing
set at a cut is the running activation plus every backbone tap already produced —
the graph-cut rule — so multi-tap heads (LRASPP's low/high) are priced honestly
rather than under-counted. Cutting after the final backbone child is not offered:
that is full-onboard execution wearing a costume.

Head and tail together execute exactly the operations the full model executes (taps
collected on whichever side produces them; the inference-only ``aux`` head is never
run), so a ``float32`` cut reproduces the full model's output bit-for-bit and the
only quality effect comes from the declared feature reduction. Requires the
``[v2-real-models]`` extra; torch is imported lazily via the shared backend cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.real_models import TorchSegmentationBackend, backend_for
from aerointentbench.v2.scenario import ExecutorConfigSpec

__all__ = ["TorchSplitPartition", "legal_cuts"]


def legal_cuts(backend: TorchSegmentationBackend) -> tuple[str, ...]:
    """The backbone child names after which a split may cut (the last is excluded)."""
    names = [name for name, _ in backend._model.backbone.named_children()]
    return tuple(names[:-1])


class TorchSplitPartition:
    """Head/tail partition of one cached torchvision segmentation model."""

    def __init__(self, spec: ExecutorConfigSpec) -> None:
        params = {
            key.removeprefix("backend_"): value
            for key, value in spec.parameters.items()
            if key.startswith("backend_")
        }
        self._backend = backend_for(_as_backend_parameters(params))
        self._threshold = float(params.get("probability_threshold", 0.5))
        self._cut = str(spec.parameters["split_cut"])
        cuts = legal_cuts(self._backend)
        if self._cut not in cuts:
            raise SchemaValidationError(
                f"split config {spec.config_id!r}: split_cut {self._cut!r} is not a legal "
                f"cut of {params.get('model_id')!r}; legal cuts: {list(cuts)}"
            )
        self.model_strategy_id = f"SPLIT({self._backend.info.model_id}@{self._cut})"

    # -- head ---------------------------------------------------------------------------

    def run_head(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        backend = self._backend
        torch = backend._torch
        tensor = backend._transform(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
        batch = tensor.unsqueeze(0).to(device=backend._device, dtype=backend._dtype)

        crossing: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            x = batch
            for name, module in backend._model.backbone.named_children():
                x = module(x)
                tap = backend._model.backbone.return_layers.get(name)
                if tap is not None:
                    crossing[tap] = _to_numpy(x)
                if name == self._cut:
                    break
        # The running activation crosses too — unless the cut IS a tap boundary, in
        # which case the tap entry already carries it and pricing it twice would lie.
        if backend._model.backbone.return_layers.get(self._cut) is None:
            crossing["cursor"] = _to_numpy(x)
        return crossing

    # -- tail ---------------------------------------------------------------------------

    def run_tail(self, tensors: dict[str, np.ndarray], rgb_shape: tuple[int, ...]) -> np.ndarray:
        backend = self._backend
        torch = backend._torch
        to_device = {
            name: torch.from_numpy(array).to(device=backend._device, dtype=backend._dtype)
            for name, array in tensors.items()
        }
        features: dict[str, Any] = {
            name: value for name, value in to_device.items() if name != "cursor"
        }
        cut_tap = backend._model.backbone.return_layers.get(self._cut)
        with torch.inference_mode():
            x = to_device["cursor"] if cut_tap is None else to_device[cut_tap]
            passed_cut = False
            for name, module in backend._model.backbone.named_children():
                if passed_cut:
                    x = module(x)
                    tap = backend._model.backbone.return_layers.get(name)
                    if tap is not None:
                        features[tap] = x
                if name == self._cut:
                    passed_cut = True

            model = backend._model
            if type(model).__name__ == "LRASPP":  # classifier consumes the tap dict
                logits = model.classifier(features)
            else:  # _SimpleSegmentationModel: FCN / DeepLabV3 use the 'out' tap only
                logits = model.classifier(features["out"])
            logits = torch.nn.functional.interpolate(
                logits, size=rgb_shape[:2], mode="bilinear", align_corners=False
            )
            probabilities = torch.softmax(logits.float(), dim=1)[0, backend._class_index]
        probability_map = probabilities.cpu().numpy().astype(np.float32)
        return probability_map >= self._threshold


def _to_numpy(tensor: Any) -> np.ndarray:
    return tensor.float().cpu().numpy().astype(np.float32)


def _as_backend_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    required = ("model_id", "weights_id", "task_class", "input_width_px", "input_height_px")
    missing = [key for key in required if key not in params]
    if missing:
        raise SchemaValidationError(
            f"simulated_split torch backend parameters missing {missing} "
            f"(declare them as backend_<name>)"
        )
    return {
        "model_id": params["model_id"],
        "weights_id": params["weights_id"],
        "task_class": params["task_class"],
        "device": params.get("device", "auto"),
        "dtype": params.get("dtype", "float32"),
        "input_width_px": params["input_width_px"],
        "input_height_px": params["input_height_px"],
        "warmup_runs": params.get("warmup_runs", 1),
    }
