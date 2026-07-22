"""Execution backends: how a selected configuration actually produces a result.

Responsibility
--------------
Turn an ``ExecutionRequest`` (episode, frame, selected configuration, current network
observation, time, seed) into an ``ExecutionResult`` (success, latency, energy,
communication volume, prediction payload, failure reason, metadata).

Modules
-------
- ``base``                      -- ``Executor`` and ``PredictionSource`` protocols,
                                   ``ExecutionRequest``/``ExecutionResult``, ``FailureReason``.
- ``profile_executor``          -- costs synthesised from a per-platform profile, including
                                   the remote latency model. The V1 default.
- ``replay_executor``           -- precomputed outcomes keyed by frame x configuration.
- ``real_segmentation_executor``-- documented stub; no ML dependency in V1.
- ``registry``                  -- name-to-backend mapping for the composition root.

Boundaries
----------
The runner must not care which backend produced a result. Backends are selected through the
registry, never through ``if executor_type == ...`` in the loop. Behaviour comes from typed
configuration metadata -- ``strategy.placement`` decides local versus remote -- and never
from parsing a ``config_id``.

Predictions are opaque here. Their shape belongs to the task; an executor that understood
them would grow a branch per task.
"""

from aerointentbench.executor.base import (
    ExecutionRequest,
    ExecutionResult,
    Executor,
    FailureReason,
    Prediction,
    PredictionSource,
)
from aerointentbench.executor.profile_executor import (
    DEFAULT_REMOTE_TIMEOUT_S,
    ProfileExecutor,
    remote_latency_s,
    transfer_time_s,
)
from aerointentbench.executor.real_segmentation_executor import RealSegmentationExecutor
from aerointentbench.executor.registry import executor_registry
from aerointentbench.executor.replay_executor import (
    ReplayExecutor,
    ReplayRecord,
    load_replay_records,
    make_replay_executor,
)

__all__ = [
    "DEFAULT_REMOTE_TIMEOUT_S",
    "ExecutionRequest",
    "ExecutionResult",
    "Executor",
    "FailureReason",
    "Prediction",
    "PredictionSource",
    "ProfileExecutor",
    "RealSegmentationExecutor",
    "ReplayExecutor",
    "ReplayRecord",
    "executor_registry",
    "load_replay_records",
    "make_replay_executor",
    "remote_latency_s",
    "transfer_time_s",
]
