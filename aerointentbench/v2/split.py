"""Split inference: the head runs onboard, intermediate features cross the link.

The third deployment option next to full-onboard and full-server: the drone runs the
model's first stages (the *head*), transmits the activation tensors crossing the cut
(the *features*), and a server runs the remainder (the *tail*). What this module owns:

- the **graph-cut payload rule**: the payload is every tensor crossing the cut — the
  running activation plus any earlier skip/tap tensors the tail still needs — with the
  size computed from the actual arrays, never configured;
- **feature reduction** (``float32`` / ``float16`` / ``uint8`` per-tensor affine): the
  reduction is applied to the tensors the tail actually consumes, so its accuracy cost
  is real, not narrated;
- the ``SimulatedSplitExecutor``: a :class:`SimulatedRemoteExecutor` subclass that only
  changes *what* is sent (via ``_prepare_payload``) — transport rules, failure
  precedence, upload-charged-on-failure, and fallback semantics stay in one place.

Privacy: a split configuration transmits features, not raw input, and the catalog maps
it to ``transmitted_payload="features"`` — legal under ``features_only`` where raw-RGB
offload is forbidden. Compression is not de-identification for images; features are
*declared* non-invertible, which is the same trust boundary the V1 privacy model
already documents.

The real torch partition lives in :mod:`aerointentbench.v2.split_models` (optional
``[v2-real-models]`` extra); the heuristic partition here is a CI test fixture, never
performance evidence. Learned feature compression (bottleneck autoencoders) is
explicitly out of scope — it requires training pipelines this benchmark bans.
"""

from __future__ import annotations

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
    "FEATURE_DTYPES",
    "SPLIT_PARAMETER_DEFAULTS",
    "SimulatedSplitExecutor",
    "SplitPartition",
    "build_split_executor",
    "encode_features",
]

FEATURE_DTYPES = ("float32", "float16", "uint8")

#: Split-specific defaults on top of the shared remote transport defaults.
SPLIT_PARAMETER_DEFAULTS = {
    **REMOTE_PARAMETER_DEFAULTS,
    "feature_dtype": "float16",
}


class SplitPartition(Protocol):
    """One model cut into an onboard head and a server tail.

    ``run_head`` returns every float32 tensor crossing the cut, keyed by role (the
    running activation is ``"cursor"``; taps keep their tap names). ``run_tail``
    consumes the (possibly reduced) tensors plus the capture shape and returns the
    final boolean prediction mask at capture resolution.
    """

    model_strategy_id: str

    def run_head(self, rgb: np.ndarray) -> dict[str, np.ndarray]: ...

    def run_tail(
        self, tensors: dict[str, np.ndarray], rgb_shape: tuple[int, ...]
    ) -> np.ndarray: ...


def encode_features(
    tensors: dict[str, np.ndarray], dtype: str
) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    """Reduce the crossing tensors to ``dtype`` and price the payload from real bytes.

    Returns ``(dequantized, payload_mb, detail)``: the tensors the tail will actually
    consume (reduction round-tripped, so its accuracy cost is real), the exact wire
    size in MB, and a per-tensor breakdown for diagnostics. ``uint8`` uses per-tensor
    affine min/max quantization; scale metadata is a few bytes and not priced.
    """
    if dtype not in FEATURE_DTYPES:
        raise SchemaValidationError(
            f"feature_dtype {dtype!r} must be one of {list(FEATURE_DTYPES)}"
        )
    dequantized: dict[str, np.ndarray] = {}
    detail: dict[str, Any] = {}
    total_bytes = 0
    for name, array in tensors.items():
        source = array.astype(np.float32, copy=False)
        if dtype == "float32":
            wire = source
            restored = source
        elif dtype == "float16":
            wire = source.astype(np.float16)
            restored = wire.astype(np.float32)
        else:  # uint8 per-tensor affine
            low = float(source.min()) if source.size else 0.0
            high = float(source.max()) if source.size else 0.0
            scale = (high - low) / 255.0 or 1.0
            wire = np.clip(np.round((source - low) / scale), 0, 255).astype(np.uint8)
            restored = wire.astype(np.float32) * scale + low
        total_bytes += wire.nbytes
        detail[name] = {"shape": list(source.shape), "wire_bytes": int(wire.nbytes)}
        dequantized[name] = restored
    return dequantized, total_bytes / 1e6, detail


class _HeuristicSplitPartition:
    """CI test fixture: 2x-downsampled frame as the "features", a heuristic tail.

    Exists so schema/privacy/transport/accounting semantics are testable without torch.
    Its "features" are trivially image-like, so it must never be cited as evidence
    about feature privacy or split quality — it is a fixture, like the heuristic
    executors it wraps.
    """

    def __init__(self, tail_executor: ImageExecutor) -> None:
        self._tail = tail_executor
        self.model_strategy_id = f"SPLIT({tail_executor.model_strategy_id})"

    def run_head(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        return {"cursor": rgb[::2, ::2].astype(np.float32)}

    def run_tail(self, tensors: dict[str, np.ndarray], rgb_shape: tuple[int, ...]) -> np.ndarray:
        cursor = tensors["cursor"]
        upsampled = np.repeat(np.repeat(cursor, 2, axis=0), 2, axis=1)
        reconstructed = np.clip(upsampled[: rgb_shape[0], : rgb_shape[1]], 0, 255).astype(np.uint8)
        return self._tail.run(reconstructed).prediction_mask


class _SplitTailBackend:
    """Adapter presenting the tail as a :class:`RemoteInferenceBackend`.

    The transport hands over the in-process payload it was given; like the raw-RGB
    simulation, the payload is resolved directly rather than serialized, so the actual
    tensors travel via the executor's per-call stash (single-threaded, set immediately
    before the transport runs).
    """

    def __init__(self, partition: SplitPartition) -> None:
        self._partition = partition
        self.model_strategy_id = f"TAIL({partition.model_strategy_id})"
        self.stashed_tensors: dict[str, np.ndarray] | None = None
        self.stashed_shape: tuple[int, ...] | None = None

    def infer(self, request: object, rgb: np.ndarray) -> np.ndarray:
        del request, rgb
        assert self.stashed_tensors is not None and self.stashed_shape is not None
        try:
            return self._partition.run_tail(self.stashed_tensors, self.stashed_shape)
        finally:
            self.stashed_tensors = None
            self.stashed_shape = None


class SimulatedSplitExecutor(SimulatedRemoteExecutor):
    """Head onboard, features across the simulated link, tail on the simulated server.

    Cost semantics: the head's configured latency (``head_latency_s``) and the config's
    ``energy_j_per_call`` (the head's compute energy) are charged on EVERY attempt —
    including failed uploads, where the head work is genuinely wasted. Payload MB come
    from the tensors actually produced this call; transfer energy uses the shared
    configured J/MB radio model.
    """

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        transport: SimulatedTransport,
        partition: SplitPartition,
        tail_backend: _SplitTailBackend,
        *,
        fallback: ImageExecutor | None = None,
    ) -> None:
        super().__init__(spec, transport, fallback=fallback)
        self._partition = partition
        self._tail_backend = tail_backend
        self._params = {**SPLIT_PARAMETER_DEFAULTS, **dict(spec.parameters)}

    def _prepare_payload(self, rgb: np.ndarray) -> PreparedPayload:
        crossing = self._partition.run_head(rgb)
        dtype = str(self._params["feature_dtype"])
        dequantized, payload_mb, detail = encode_features(crossing, dtype)
        self._tail_backend.stashed_tensors = dequantized
        self._tail_backend.stashed_shape = tuple(rgb.shape)
        # Wire-visible shape: the running activation, or the largest crossing tensor
        # when the cut coincides with a tap boundary (no separate cursor then).
        principal = (
            crossing["cursor"]
            if "cursor" in crossing
            else max(crossing.values(), key=lambda array: array.nbytes)
        )
        cursor_shape = tuple(int(v) for v in principal.shape)
        return PreparedPayload(
            kind="features",
            shape=cursor_shape,
            mb=payload_mb,
            payload=rgb,  # unused by the tail; the stash carries the real tensors
            onboard_latency_s=float(self._params["head_latency_s"]),
            onboard_energy_j=self._spec.energy_j_per_call,
            diagnostics={
                "execution_location": "split",
                "split_cut": str(self._params["split_cut"]),
                "feature_dtype": dtype,
                "feature_payload": detail,
                "feature_payload_mb": payload_mb,
            },
        )


def build_split_executor(
    spec: ExecutorConfigSpec, local_registry: dict[str, ImageExecutor]
) -> SimulatedSplitExecutor:
    """Wire a split executor from its spec and the already-built local executors.

    ``backend_kind`` selects the partition: ``torch_semantic_segmentation`` cuts the
    real model (optional extra); the heuristic kinds produce the CI fixture partition.
    The optional ``fallback_config_id`` must name an already-built local executor.
    """
    params = {**SPLIT_PARAMETER_DEFAULTS, **dict(spec.parameters)}
    backend_kind = str(spec.parameters["backend_kind"])

    partition: SplitPartition
    if backend_kind == "torch_semantic_segmentation":
        from aerointentbench.v2.split_models import TorchSplitPartition

        partition = TorchSplitPartition(spec)
    elif backend_kind in ("fast_weak", "slow_strong"):
        from aerointentbench.v2.executors import build_executors
        from aerointentbench.v2.scenario import ExecutorConfigSpec as Spec

        tail_spec = Spec(
            config_id=f"{spec.config_id}__tail",
            model_strategy_id=f"{spec.model_strategy_id}__tail",
            kind=backend_kind,
            mission_latency_s=float(params["remote_compute_s"]),
            energy_j_per_call=0.0,
            communication_mb_per_call=0.0,
            quality_tier=spec.quality_tier,
            parameters={
                key.removeprefix("backend_"): value
                for key, value in spec.parameters.items()
                if key.startswith("backend_") and key != "backend_kind"
            },
        )
        partition = _HeuristicSplitPartition(build_executors((tail_spec,))[tail_spec.config_id])
    else:
        raise SchemaValidationError(
            f"split backend_kind {backend_kind!r} must be a local executor kind"
        )

    tail_backend = _SplitTailBackend(partition)
    transport = SimulatedTransport(params, tail_backend)

    fallback = None
    fallback_id = spec.parameters.get("fallback_config_id")
    if fallback_id is not None:
        if str(fallback_id) not in local_registry:
            raise SchemaValidationError(
                f"split config {spec.config_id!r}: fallback_config_id {fallback_id!r} does not "
                f"name an already-built local executor"
            )
        fallback = local_registry[str(fallback_id)]

    return SimulatedSplitExecutor(spec, transport, partition, tail_backend, fallback=fallback)
