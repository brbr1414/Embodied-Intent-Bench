"""Pre-split models: literature-defined split configurations as catalog options.

The ``pretrained_split`` executor kind runs a model that was *published already
split*: an onboard head (encoder) and a server tail trained together by prior
split-computing research, distributed as pre-trained checkpoints. This is the
answer to "where should the network be split?" that this benchmark deliberately
does NOT research — the split point is a fixed property of a catalog entry, taken
from the source paper, never searched, tuned, or optimized here. What the
benchmark studies is the *selection* between such configurations under a mission
contract.

Boundaries this module enforces:

- **Provenance is mandatory** (:class:`SplitSpec`): every ``pretrained_split``
  config must name where its split came from (``source`` = paper /
  official_repository, a citable ``source_reference``, and the ``split_location``
  in the source's terms). A split without provenance fails at scenario load.
- **Payload honesty**: the wire size is the byte length of what the head actually
  encodes for this frame (for the sc2 backend: the entropy-coded bitstream), never
  a configured constant. ``communication_mb_per_call`` remains the config's
  nominal/documentation value.
- **One transport**: the executor subclasses :class:`SimulatedRemoteExecutor`
  through the same ``_prepare_payload`` hook as the graph-cut split — stage
  timing, failure precedence, upload-charged-on-failure, and same-frame fallback
  stay the single documented implementation. Head latency (configured, pending
  board measurement) and head energy (``energy_j_per_call``) are charged on every
  attempt, including failed uploads.
- **Privacy**: the catalog maps this kind to REMOTE placement with
  ``transmitted_payload="features"`` — the payload is a learned compressed
  representation, not raw RGB. Same declared-non-invertible trust boundary as the
  graph-cut split.

The genuine backend (``sc2_entropic_student``) consumes checkpoints from the
SC2 benchmark (Matsubara et al., "SC2 Benchmark: Supervised Compression for Split
Computing", TMLR 2023; Entropic Student from Matsubara et al., WACV 2022). Its
heavy dependencies (torch + the ``sc2bench`` package) are imported lazily and live
in the optional ``[v2-presplit]`` extra; the default suite uses the labelled
fixture backend and downloads nothing. Training or fine-tuning bottlenecks
ourselves remains banned — checkpoints are consumed exactly like torchvision
weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.v2.executors import ImageExecutor
from aerointentbench.v2.remote import (
    REMOTE_PARAMETER_DEFAULTS,
    PreparedPayload,
    SimulatedRemoteExecutor,
    SimulatedTransport,
)
from aerointentbench.v2.scenario import ExecutorConfigSpec

__all__ = [
    "PRESPLIT_BACKENDS",
    "PRESPLIT_PARAMETER_DEFAULTS",
    "PresplitModel",
    "PretrainedSplitExecutor",
    "SplitSpec",
    "build_presplit_executor",
]

#: Where a predefined split may come from. There is intentionally no value that
#: could describe a split this benchmark discovered itself.
SPLIT_SOURCES = ("paper", "official_repository")

PRESPLIT_BACKENDS = ("sc2_entropic_student", "fixture")

PRESPLIT_PARAMETER_DEFAULTS = dict(REMOTE_PARAMETER_DEFAULTS)


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """The provenance of a predefined split — why THIS cut, answered by citation.

    ``split_location`` uses the source's own vocabulary (e.g. "bottleneck replaces
    conv1..layer1 of ResNet-50"); it is documentation, not an instruction this
    benchmark interprets — the checkpoint already embodies the split.
    """

    split_id: str
    split_location: str
    source: str
    source_reference: str

    @classmethod
    def from_parameters(cls, config_id: str, parameters: dict[str, Any]) -> SplitSpec:
        source = str(parameters["split_source"])
        if source not in SPLIT_SOURCES:
            raise SchemaValidationError(
                f"config {config_id!r}: split_source {source!r} must be one of "
                f"{list(SPLIT_SOURCES)} — a predefined split is adopted from prior work, "
                "never discovered by this benchmark"
            )
        return cls(
            split_id=config_id,
            split_location=str(parameters["split_location"]),
            source=source,
            source_reference=str(parameters["split_source_reference"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "split_id": self.split_id,
            "split_location": self.split_location,
            "source": self.source,
            "source_reference": self.source_reference,
        }


class PresplitModel(Protocol):
    """One published pre-split model: head encode onboard, tail completion remote.

    ``encode`` returns ``(payload, payload_mb, diagnostics)`` where ``payload`` is
    an opaque object the tail consumes and ``payload_mb`` is priced from the real
    encoded bytes. ``complete`` consumes the payload plus the capture shape and
    returns the final boolean mask at capture resolution.
    """

    model_strategy_id: str

    def encode(self, rgb: np.ndarray) -> tuple[Any, float, dict[str, Any]]: ...

    def complete(self, payload: Any, rgb_shape: tuple[int, ...]) -> np.ndarray: ...


class _FixturePresplitModel:
    """CI test fixture: 4x-downsampled grayscale bytes as the "bitstream".

    Exists so schema/provenance/privacy/accounting semantics are testable without
    torch or sc2bench. Its payload is trivially image-derived, so it must never be
    cited as evidence about feature privacy, payload size, or split quality.
    """

    def __init__(self, tail_executor: ImageExecutor) -> None:
        self._tail = tail_executor
        self.model_strategy_id = f"PRESPLIT_FIXTURE({tail_executor.model_strategy_id})"

    def encode(self, rgb: np.ndarray) -> tuple[Any, float, dict[str, Any]]:
        gray = rgb.mean(axis=2)[::4, ::4].astype(np.uint8)
        payload = gray.tobytes()
        return (
            {"gray": gray, "shape": rgb.shape},
            len(payload) / 1e6,
            {"payload_bytes": len(payload), "fixture": True},
        )

    def complete(self, payload: Any, rgb_shape: tuple[int, ...]) -> np.ndarray:
        gray = payload["gray"]
        upsampled = np.repeat(np.repeat(gray, 4, axis=0), 4, axis=1)[: rgb_shape[0], : rgb_shape[1]]
        rgb = np.stack([upsampled] * 3, axis=2).astype(np.uint8)
        return self._tail.run(rgb).prediction_mask


# --- sc2 backend (lazy heavy imports; [v2-presplit] extra) -----------------------------------

_SC2_REQUIRED_PARAMETERS = (
    "checkpoint_path",
    "input_width_px",
    "input_height_px",
    "device",
)

#: Process-wide cache so one strategy loads its checkpoint at most once (mirrors
#: the real-model backend cache; keyed by everything that changes the artefact).
_SC2_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class _Sc2EntropicStudentModel:
    """Entropic Student DeepLabV3-R50 (VOC) behind the :class:`PresplitModel` seam.

    The checkpoint IS the split: ``backbone.bottleneck_layer.encoder`` (the head
    the paper trains for the device) plus the entropy coder produce the wire
    bitstream; ``complete`` entropy-decodes and runs layer2..4 + classifier (the
    server side). The person class index is resolved from torchvision's VOC-label
    weight metadata — the same 21-class label set the checkpoint was trained on —
    never hardcoded.
    """

    def __init__(self, spec: ExecutorConfigSpec) -> None:
        params = dict(spec.parameters)
        missing = [key for key in _SC2_REQUIRED_PARAMETERS if key not in params]
        if missing:
            raise SchemaValidationError(
                f"config {spec.config_id!r}: sc2_entropic_student requires parameters "
                f"{list(_SC2_REQUIRED_PARAMETERS)}; missing {missing}"
            )
        self._spec = spec
        self._width = int(params["input_width_px"])
        self._height = int(params["input_height_px"])
        self._device = str(params["device"])
        self._checkpoint_path = str(params["checkpoint_path"])
        self.model_strategy_id = spec.model_strategy_id
        self._model = self._load(self._checkpoint_path, self._device)
        self._person_index = self._resolve_person_index()

    @staticmethod
    def _load(checkpoint_path: str, device: str) -> Any:
        from pathlib import Path

        checkpoint_path = str(Path(checkpoint_path).expanduser())
        key = (checkpoint_path, device)
        if key in _SC2_MODEL_CACHE:
            return _SC2_MODEL_CACHE[key]
        try:
            import torch
            from sc2bench.models.segmentation.registry import get_segmentation_model
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise SchemaValidationError(
                "a pretrained_split config with split_backend 'sc2_entropic_student' needs "
                "the [v2-presplit] extra (sc2bench + torch); install it or use the replay/"
                "fixture path"
            ) from error

        model = get_segmentation_model(
            "deeplabv3_model",
            pretrained=False,
            pretrained_backbone_name="resnet50",
            num_classes=21,
            uses_aux=True,
            num_input_channels=2048,
            num_aux_channels=1024,
            return_layer_dict={"layer3": "aux", "layer4": "out"},
            analyzable_layer_key="bottleneck_layer",
            backbone_config={
                "key": "splittable_resnet",
                "kwargs": {
                    "num_classes": 1000,
                    "pretrained": False,
                    "replace_stride_with_dilation": [False, True, True],
                    "bottleneck_config": {
                        "key": "FPBasedResNetBottleneck",
                        "kwargs": {"num_bottleneck_channels": 24, "num_target_channels": 256},
                    },
                    "resnet_name": "resnet50",
                    "pre_transform": None,
                    "skips_avgpool": True,
                    "skips_fc": True,
                },
            },
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        has_model_key = isinstance(checkpoint, dict) and "model" in checkpoint
        state = checkpoint["model"] if has_model_key else checkpoint
        # The entropy coder's CDF tables are sized by the checkpoint; shape the empty
        # buffers first so a strict load verifies every key.
        bottleneck = model.backbone.bottleneck_layer.entropy_bottleneck
        prefix = "backbone.bottleneck_layer.entropy_bottleneck."
        for name in ("_offset", "_quantized_cdf", "_cdf_length"):
            setattr(bottleneck, name, torch.empty_like(state[prefix + name]))
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        _SC2_MODEL_CACHE[key] = model
        return model

    @staticmethod
    def _resolve_person_index() -> int:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

        categories = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]
        return categories.index("person")

    def _preprocess(self, rgb: np.ndarray) -> Any:
        import torch
        from PIL import Image

        image = Image.fromarray(rgb).resize((self._width, self._height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def encode(self, rgb: np.ndarray) -> tuple[Any, float, dict[str, Any]]:
        import torch

        with torch.no_grad():
            compressed = self._model.backbone.bottleneck_layer.encode(self._preprocess(rgb))
        payload_bytes = sum(len(s) for s in compressed["strings"][0])
        return (
            compressed,
            payload_bytes / 1e6,
            {"payload_bytes": payload_bytes, "latent_shape": list(compressed["shape"])},
        )

    def complete(self, payload: Any, rgb_shape: tuple[int, ...]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        backbone = self._model.backbone
        with torch.no_grad():
            features = backbone.bottleneck_layer.decode(**payload)
            features = backbone.layer4(backbone.layer3(backbone.layer2(features)))
            logits = self._model.classifier(features)
            logits = functional.interpolate(
                logits, size=(self._height, self._width), mode="bilinear", align_corners=False
            )
            classes = logits.argmax(1)[0].cpu().numpy()
        mask = (classes == self._person_index).astype(np.uint8)
        resized = Image.fromarray(mask * 255).resize((rgb_shape[1], rgb_shape[0]), Image.NEAREST)
        return np.asarray(resized) > 127


class _PresplitTailBackend:
    """Adapter presenting ``complete`` as a :class:`RemoteInferenceBackend`.

    Like the graph-cut split, the encoded payload travels via the executor's
    per-call stash (single-threaded; set immediately before the transport runs).
    """

    def __init__(self, model: PresplitModel) -> None:
        self._model = model
        self.model_strategy_id = f"TAIL({model.model_strategy_id})"
        self.stashed_payload: Any | None = None
        self.stashed_shape: tuple[int, ...] | None = None

    def infer(self, request: object, rgb: np.ndarray) -> np.ndarray:
        del request, rgb
        assert self.stashed_payload is not None and self.stashed_shape is not None
        try:
            return self._model.complete(self.stashed_payload, self.stashed_shape)
        finally:
            self.stashed_payload = None
            self.stashed_shape = None


class PretrainedSplitExecutor(SimulatedRemoteExecutor):
    """Published head onboard, its bitstream across the link, published tail remote.

    Cost semantics match the graph-cut split: configured head latency
    (``head_latency_s``, pending board measurement — the measured encode wall
    clock is recorded in diagnostics) and the config's ``energy_j_per_call`` are
    charged on every attempt, including failed uploads. Payload MB come from the
    bytes the head actually encoded for this frame.
    """

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        transport: SimulatedTransport,
        model: PresplitModel,
        tail_backend: _PresplitTailBackend,
        split_spec: SplitSpec,
        *,
        fallback: ImageExecutor | None = None,
    ) -> None:
        super().__init__(spec, transport, fallback=fallback)
        self._model = model
        self._tail_backend = tail_backend
        self._split_spec = split_spec
        self._params = {**PRESPLIT_PARAMETER_DEFAULTS, **dict(spec.parameters)}

    @property
    def split_spec(self) -> SplitSpec:
        return self._split_spec

    def _prepare_payload(self, rgb: np.ndarray) -> PreparedPayload:
        # The head's wall clock is not recorded here: mission results must stay
        # byte-deterministic, and head_latency_s is declared configured. Board
        # measurement is the empirical follow-up, not a per-run diagnostic.
        payload, payload_mb, encode_diagnostics = self._model.encode(rgb)
        self._tail_backend.stashed_payload = payload
        self._tail_backend.stashed_shape = tuple(rgb.shape)
        return PreparedPayload(
            kind="features",
            shape=tuple(rgb.shape),
            mb=payload_mb,
            payload=rgb,  # unused by the tail; the stash carries the encoded payload
            onboard_latency_s=float(self._params["head_latency_s"]),
            onboard_energy_j=self._spec.energy_j_per_call,
            diagnostics={
                "execution_location": "split",
                "split_spec": self._split_spec.to_dict(),
                "feature_payload_mb": payload_mb,
                **encode_diagnostics,
            },
        )


def build_presplit_executor(
    spec: ExecutorConfigSpec, local_registry: dict[str, ImageExecutor]
) -> PretrainedSplitExecutor:
    """Wire a pretrained-split executor from its spec and the built local executors.

    ``split_backend`` selects the model: ``sc2_entropic_student`` loads the real
    checkpoint (optional extra); ``fixture`` wraps a local heuristic kind
    (``backend_kind``) for CI. The optional ``fallback_config_id`` must name an
    already-built local executor.
    """
    params = {**PRESPLIT_PARAMETER_DEFAULTS, **dict(spec.parameters)}
    split_spec = SplitSpec.from_parameters(spec.config_id, dict(spec.parameters))
    backend = str(spec.parameters["split_backend"])

    model: PresplitModel
    if backend == "sc2_entropic_student":
        model = _Sc2EntropicStudentModel(spec)
    elif backend == "fixture":
        from aerointentbench.v2.executors import build_executors
        from aerointentbench.v2.scenario import ExecutorConfigSpec as Spec

        tail_kind = str(spec.parameters.get("backend_kind", "fast_weak"))
        if tail_kind not in ("fast_weak", "slow_strong"):
            raise SchemaValidationError(
                f"config {spec.config_id!r}: fixture backend_kind {tail_kind!r} must be a "
                "heuristic local kind"
            )
        tail_spec = Spec(
            config_id=f"{spec.config_id}__tail",
            model_strategy_id=f"{spec.model_strategy_id}__tail",
            kind=tail_kind,
            mission_latency_s=float(params["remote_compute_s"]),
            energy_j_per_call=0.0,
            communication_mb_per_call=0.0,
            quality_tier=spec.quality_tier,
            parameters={},
        )
        model = _FixturePresplitModel(build_executors((tail_spec,))[tail_spec.config_id])
    else:
        raise SchemaValidationError(
            f"config {spec.config_id!r}: split_backend {backend!r} must be one of "
            f"{list(PRESPLIT_BACKENDS)}"
        )

    tail_backend = _PresplitTailBackend(model)
    transport = SimulatedTransport(params, tail_backend)

    fallback = None
    fallback_id = spec.parameters.get("fallback_config_id")
    if fallback_id is not None:
        if str(fallback_id) not in local_registry:
            raise SchemaValidationError(
                f"presplit config {spec.config_id!r}: fallback_config_id {fallback_id!r} does "
                "not name an already-built local executor"
            )
        fallback = local_registry[str(fallback_id)]

    return PretrainedSplitExecutor(
        spec, transport, model, tail_backend, split_spec, fallback=fallback
    )
