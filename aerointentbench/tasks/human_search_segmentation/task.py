"""Binds the human-search tracker and evaluator into one registered task."""

from __future__ import annotations

from pathlib import Path

from aerointentbench.schemas.episode import Episode
from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.schemas.profile import ProfileCatalog
from aerointentbench.schemas.task_spec import TaskSpec
from aerointentbench.tasks.human_search_segmentation.evaluator import (
    HumanSearchSegmentationEvaluator,
)
from aerointentbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceTracker,
)
from aerointentbench.tasks.human_search_segmentation.ground_truth import (
    TASK_ID,
    HumanSearchGroundTruth,
    load_human_search_ground_truth,
)
from aerointentbench.tasks.human_search_segmentation.prediction import (
    SyntheticHumanSearchPredictions,
)

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

    def load_ground_truth(
        self, directory: Path, episode: Episode
    ) -> HumanSearchGroundTruth | None:
        """Find the answers for this episode's *frame stream*.

        Keyed on the stream, not the episode, so several episodes flying the same scene under
        different battery, network, or contract conditions are scored identically. Every file
        in the directory is read rather than a filename convention being assumed, so renaming
        a fixture cannot silently unhook it from its episodes.
        """
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json")):
            ground_truth = load_human_search_ground_truth(path)
            if ground_truth.frame_stream_id == episode.frame_stream_id:
                return ground_truth
        return None

    def create_prediction_source(
        self, ground_truth: HumanSearchGroundTruth | None, profiles: ProfileCatalog
    ) -> SyntheticHumanSearchPredictions | None:
        """Return the synthetic prediction source, or ``None`` without answers to draw from.

        The source reads ground truth, which is legitimate -- it stands in for a model that
        would actually be looking at the scene. What it must never do is let a ground-truth
        quantity reach a policy, and it does not: the hidden fields it writes are excluded
        from the tracker's policy summary.
        """
        if ground_truth is None:
            return None
        return SyntheticHumanSearchPredictions(ground_truth, profiles)
