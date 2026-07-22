"""Binds the human-search tracker and evaluator into one registered task."""

from __future__ import annotations

from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.schemas.task_spec import TaskSpec
from aerointentbench.tasks.human_search_segmentation.evaluator import (
    HumanSearchSegmentationEvaluator,
)
from aerointentbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceTracker,
)
from aerointentbench.tasks.human_search_segmentation.ground_truth import TASK_ID

__all__ = ["HumanSearchSegmentationTask"]


class HumanSearchSegmentationTask:
    """The V1 task, resolved from the registry by ``contract.task_id``."""

    __slots__ = ("_task_spec",)

    def __init__(self, task_spec: TaskSpec) -> None:
        if task_spec.task_id != TASK_ID:
            raise SchemaValidationError(
                f"{type(self).__name__} implements {TASK_ID!r}, "
                f"but was given a specification for {task_spec.task_id!r}"
            )
        self._task_spec = task_spec

    @property
    def task_id(self) -> str:
        return TASK_ID

    @property
    def task_spec(self) -> TaskSpec:
        return self._task_spec

    def create_tracker(self) -> HumanSearchEvidenceTracker:
        """Return a fresh tracker. Trackers are stateful, so one per episode."""
        return HumanSearchEvidenceTracker()

    def create_evaluator(self) -> HumanSearchSegmentationEvaluator:
        return HumanSearchSegmentationEvaluator(self._task_spec)
