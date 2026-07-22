"""The executor registry.

Lets the composition root pick a backend by name -- ``--executor profile`` -- without the
runner importing any backend, and lets a new backend join by registering itself rather than
by editing a branch somewhere in the core.

Registration happens here, explicitly, rather than through import scanning: a reader should
be able to see every available backend in one place.
"""

from __future__ import annotations

from aerointentbench.executor.base import Executor
from aerointentbench.executor.profile_executor import ProfileExecutor
from aerointentbench.executor.real_segmentation_executor import RealSegmentationExecutor
from aerointentbench.executor.replay_executor import make_replay_executor
from aerointentbench.registry import Registry

__all__ = ["executor_registry"]

executor_registry: Registry[Executor] = Registry("executor")

#: Costs synthesised from a per-platform profile. The V1 default: no GPU, no weights, and
#: exactly reproducible.
executor_registry.register("profile", ProfileExecutor)

#: Precomputed outcomes replayed from a record set, so two policies that select the same
#: configuration on the same frame see identical predictions.
executor_registry.register("replay", make_replay_executor)

#: Interface stub. Registered so that selecting it fails with an explanation rather than
#: with "unknown executor", which would suggest a typo instead of a missing build.
executor_registry.register("real_segmentation", RealSegmentationExecutor)
