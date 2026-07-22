"""The deterministic episode simulator: the benchmark core.

Responsibility
--------------
Orchestrate the one-second decision loop and produce a structured episode record.
This package is the *only* part of the benchmark that owns the loop, and it must stay
free of task-specific logic.

Planned modules (added in ``feature/v1-simulator-core``)
--------------------------------------------------------
- ``episode_runner``  -- the loop; depends on protocols, never on concrete task code.
- ``state_manager``   -- assembles the policy-visible ``RuntimeState`` from internal state.
- ``battery_model``   -- ``BatteryModel`` protocol + ``SimpleBatteryModel`` (joule-based).
- ``network_trace``   -- ``NetworkModel`` protocol + ``TraceBasedNetworkModel`` lookup.
- ``action_validator``-- validates the returned config ID against the allowed pool,
                         the catalog, and the contract's privacy level.
- ``termination``     -- ``TerminationCondition`` checks (path complete, deadline, battery).

Hard rule
---------
``EpisodeRunner`` must not import a task evaluator, contain mask/IoU logic, branch on
``task_id``, or parse meaning out of configuration ID strings. Task behaviour arrives
through the ``TaskDefinition`` resolved from the task registry; resource behaviour
arrives through injected models. Adding a second task must not touch this package.
"""
