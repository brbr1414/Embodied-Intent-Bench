"""V1 task: ``HUMAN_SEARCH_SEGMENTATION``.

Find unique people along a predefined UAV path and report an instance mask set. A
target counts as found when a predicted instance matches a ground-truth instance under
the task specification's matching rule (V1: ``mask_iou >= 0.50``), deduplicated by
ground-truth track ID. Supported quality metrics: target precision, target recall,
target F1.

Planned modules (added in ``feature/v1-human-search-task``)
-----------------------------------------------------------
- ``evidence_tracker`` -- ``HumanSearchEvidenceTracker``: processed frames, accumulated
                          predicted instances, deduplicated unique-target count,
                          confidence summary, raw prediction references for the evaluator.
- ``evaluator``        -- ``HumanSearchSegmentationEvaluator``: precision/recall/F1
                          against hidden ground truth.

Boundaries
----------
The tracker's ``policy_summary()`` must expose no ground-truth-derived quantity: no
true recall, no hidden target count, no match outcome. Only ``final_record()``, consumed
by the evaluator, may reference matched ground-truth track IDs -- and that record never
reaches a policy.
"""
