"""Scores human-search evidence against hidden ground truth.

A predicted instance is a **match** when it corresponds to a real target *and* segments it
well enough to count, judged by the task specification's matching rule (V1: mask IoU at or
above 0.50). Detecting something is not the same as detecting it well enough -- which is why
a weak configuration can see a person and still fail to find them.

Counting is per **unique target**, not per instance. A mission that spots the same person on
sixty consecutive frames has found one person, so:

- a ground-truth target counts as found if *any* prediction matched it;
- a predicted identity counts as a false positive if *none* of its instances matched anything.

Both figures come out of the same pass, so precision and recall cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from aerointentbench.schemas.contract import Contract
from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.schemas.task_spec import TaskSpec
from aerointentbench.tasks.base import TaskEvaluationResult
from aerointentbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceRecord,
)
from aerointentbench.tasks.human_search_segmentation.ground_truth import HumanSearchGroundTruth

__all__ = ["HumanSearchSegmentationEvaluator", "QualityScores"]

METRIC_PRECISION: Final = "target_precision"
METRIC_RECALL: Final = "target_recall"
METRIC_F1: Final = "target_f1"


@dataclass(frozen=True, slots=True)
class QualityScores:
    """All three quality metrics plus the counts they came from."""

    target_precision: float
    target_recall: float
    target_f1: float
    matched_targets: int
    total_targets: int
    predicted_targets: int
    false_positive_targets: int

    def value_of(self, metric_name: str) -> float:
        try:
            return {
                METRIC_PRECISION: self.target_precision,
                METRIC_RECALL: self.target_recall,
                METRIC_F1: self.target_f1,
            }[metric_name]
        except KeyError:
            raise SchemaValidationError(
                f"human search cannot compute quality metric {metric_name!r}; "
                f"supported metrics are "
                f"{[METRIC_PRECISION, METRIC_RECALL, METRIC_F1]}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        return {
            METRIC_PRECISION: self.target_precision,
            METRIC_RECALL: self.target_recall,
            METRIC_F1: self.target_f1,
            "matched_targets": self.matched_targets,
            "total_targets": self.total_targets,
            "predicted_targets": self.predicted_targets,
            "false_positive_targets": self.false_positive_targets,
        }


class HumanSearchSegmentationEvaluator:
    """Turns evidence plus ground truth into a standardised task result."""

    __slots__ = ("_task_spec",)

    def __init__(self, task_spec: TaskSpec) -> None:
        self._task_spec = task_spec

    def evaluate(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchGroundTruth,
        contract: Contract,
    ) -> TaskEvaluationResult:
        scores = self.score(evidence, ground_truth)
        return TaskEvaluationResult.against_contract(
            contract,
            value=scores.value_of(contract.quality_metric),
            details={**scores.to_dict(), "processed_frames": evidence.processed_frames},
        )

    def score(
        self, evidence: HumanSearchEvidenceRecord, ground_truth: HumanSearchGroundTruth
    ) -> QualityScores:
        """Compute all three metrics. Exposed separately so tests can assert the counts."""
        rule = self._task_spec.matching_rule

        matched_targets: set[str] = set()
        predicted_identities: set[str] = set()
        matched_identities: set[str] = set()

        for instance in evidence.instances:
            predicted_identities.add(instance.predicted_target_id)
            if instance.ground_truth_track_id is None:
                continue
            if not rule.matches(instance.mask_iou):
                # Seen, but not segmented well enough to count as found.
                continue
            matched_targets.add(instance.ground_truth_track_id)
            matched_identities.add(instance.predicted_target_id)

        total_targets = len(ground_truth)
        false_positives = len(predicted_identities - matched_identities)

        recall = _ratio(len(matched_targets), total_targets, empty=1.0)
        precision = _ratio(len(matched_identities), len(predicted_identities), empty=1.0)
        return QualityScores(
            target_precision=precision,
            target_recall=recall,
            target_f1=_harmonic_mean(precision, recall),
            matched_targets=len(matched_targets),
            total_targets=total_targets,
            predicted_targets=len(predicted_identities),
            false_positive_targets=false_positives,
        )


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    """Divide, with an explicit answer for the empty case.

    An episode with no targets cannot miss any, and one that predicted nothing has produced
    no false positives -- both are vacuously perfect, so ``empty`` is 1.0 at both call sites.
    Making it a parameter keeps that choice visible instead of buried in a guard.
    """
    return empty if denominator == 0 else numerator / denominator


def _harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
