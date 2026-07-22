"""Task plugins: everything that is specific to *what the mission is measuring*.

Responsibility
--------------
Each task supplies an ``EvidenceTracker`` (accumulates evidence from execution results
and exposes a non-leaking policy summary) and a ``TaskEvaluator`` (scores the final
evidence record against hidden ground truth and returns a standardised
``TaskEvaluationResult``). A ``TaskDefinition`` binds the two together and is resolved
from the task registry by ``contract.task_id``.

V1 ships exactly one task: ``human_search_segmentation``
(added in ``feature/v1-human-search-task``).

Boundaries
----------
This is the only place mask-matching, deduplication, and target-F1 logic may live. The
simulator, metrics, and policy layers must remain unaware of it. Adding object
detection later means adding a package here plus one registry entry -- and nothing else.
"""
