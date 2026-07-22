"""Execution backends: how a selected configuration actually produces a result.

Responsibility
--------------
Turn an ``ExecutionRequest`` (episode, frame, selected configuration, current network
observation, time, seed) into an ``ExecutionResult`` (success, latency, energy,
communication volume, prediction payload, failure reason, metadata).

Planned modules (added in ``feature/v1-executors``)
---------------------------------------------------
- ``base``                      -- ``Executor`` protocol, ``ExecutionRequest``/``ExecutionResult``.
- ``profile_executor``          -- synthetic profile-driven latency/energy/communication,
                                   including the remote latency model.
- ``replay_executor``           -- serves precomputed predictions keyed by frame x config.
- ``real_segmentation_executor``-- documented stub only; no ML dependency in V1.

Boundaries
----------
The runner must not care which backend produced a result. Backends are selected
through the executor registry, never through ``if executor_type == ...`` in the loop.
Executors must not branch per task or per model ID; behaviour comes from typed
configuration metadata and profile records.
"""
