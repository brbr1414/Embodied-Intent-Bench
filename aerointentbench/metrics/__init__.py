"""Metric computation over episode records.

Responsibility
--------------
Derive per-episode metrics and cross-episode aggregates from the structured records the
simulator emits. Metrics are computed from *records*, never by reaching into live
simulator state, so a new metric can be added later without rerunning an episode as
long as the step log already carried the data.

Planned modules (added in ``feature/v1-metrics-and-cli``)
---------------------------------------------------------
- ``episode_metrics``   -- generic constraint evaluation (deadline, battery, communication,
                           privacy) plus mission success, resources, switches, violations
                           and violation margins.
- ``aggregate_metrics`` -- success rates and means across an episode suite.

Boundaries
----------
Mission-success logic consumes a standardised ``TaskEvaluationResult``
(``metric_name`` / ``value`` / ``operator`` / ``threshold`` / ``success`` / ``details``).
It must never reference a task-specific field such as ``target_f1`` by name. Adding a
task metric must not require editing this package.
"""
