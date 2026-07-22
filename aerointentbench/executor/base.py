"""The execution interface: what is asked of a backend, and what it returns.

The benchmark separates three things that are easy to conflate:

1. **what** configuration was selected -- a ``Configuration``,
2. **how** it is executed -- an :class:`Executor` implementation,
3. **what result** it produced -- an :class:`ExecutionResult`.

The runner must work identically whether a result came from a synthetic profile, a
precomputed prediction, a real local model, a remote server, or a future split-inference
runtime. Everything backend-specific stays behind this interface.

Predictions are deliberately opaque here. Their shape is a property of the *task* -- an
instance mask set for human search, boxes for detection -- and an executor that understood
them would have to grow a branch per task. The task's evidence tracker interprets the
payload; the executor and the runner only carry it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from aerointentbench.schemas.configuration import Configuration
from aerointentbench.schemas.network_trace import NetworkObservation

__all__ = [
    "ExecutionRequest",
    "ExecutionResult",
    "Executor",
    "FailureReason",
    "Prediction",
    "PredictionSource",
]

#: A task-specific prediction payload. Opaque to the executor and the runner; only the
#: task's evidence tracker interprets it.
Prediction = Any


class FailureReason(StrEnum):
    """Why an execution produced no prediction."""

    #: Remote execution was selected with no usable bandwidth.
    NETWORK_UNAVAILABLE = "network_unavailable"
    #: The transfer was possible in principle but would not have completed in time.
    TIMEOUT = "timeout"
    #: A replay backend had no record for this frame and configuration.
    NO_PREDICTION_AVAILABLE = "no_prediction_available"
    #: The backend is not implemented in this build.
    BACKEND_UNAVAILABLE = "backend_unavailable"


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Everything a backend needs to execute one configuration on one frame.

    Carries more than the V1 backends read -- ``seed`` and ``current_time_s`` are unused by
    a purely profile-driven executor -- because a request is the stable interface a future
    backend is written against, and widening it later would break every implementation.
    """

    episode_id: str
    frame_id: int
    configuration: Configuration
    network: NetworkObservation
    current_time_s: float
    #: Episode seed. A backend that introduces variation must derive it from this together
    #: with the frame and configuration, never from a global or wall-clock source, so that
    #: replaying an episode reproduces it exactly.
    seed: int


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What one execution cost, and what it produced.

    A failed execution is a normal result, not an exception. Losing the network is a
    situation the policy is meant to handle, and the vehicle keeps flying and burning
    energy while it happens -- so a failure still reports its latency and energy.
    """

    success: bool
    latency_s: float
    onboard_energy_j: float
    upload_mb: float = 0.0
    download_mb: float = 0.0
    prediction: Prediction | None = None
    failure_reason: FailureReason | None = None
    #: Backend-specific detail for the step log. Kept out of the typed fields so that a
    #: backend can record what it likes without every other backend growing the field.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.success and self.failure_reason is not None:
            raise ValueError("a successful ExecutionResult must not carry a failure_reason")
        if not self.success and self.failure_reason is None:
            raise ValueError("a failed ExecutionResult must state a failure_reason")

    @property
    def communication_mb(self) -> float:
        """Bytes actually moved, upload plus download."""
        return self.upload_mb + self.download_mb

    @property
    def latency_ms(self) -> float:
        return self.latency_s * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "latency_s": self.latency_s,
            "onboard_energy_j": self.onboard_energy_j,
            "upload_mb": self.upload_mb,
            "download_mb": self.download_mb,
            "communication_mb": self.communication_mb,
            "has_prediction": self.prediction is not None,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "metadata": dict(self.metadata),
        }


class Executor(Protocol):
    """Executes or replays one configuration on one frame."""

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Return the result of running ``request``. Must not raise on execution failure."""
        ...


class PredictionSource(Protocol):
    """Supplies the prediction a configuration would produce for a frame.

    Separated from the executor so that resource simulation and prediction generation vary
    independently: the same ``ProfileExecutor`` can run with no predictions at all (resource
    behaviour only), with a task's synthetic generator, or with recorded model output, and
    none of those choices touch the cost model.
    """

    def prediction_for(self, request: ExecutionRequest) -> Prediction | None:
        """Return the prediction for this request, or ``None`` if there is none."""
        ...
