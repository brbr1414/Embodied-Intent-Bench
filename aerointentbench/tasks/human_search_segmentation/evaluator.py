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
from aerointentbench.tasks.human_search_segmentation.ground_truth import (
    TARGET_CATEGORY,
    HumanSearchGroundTruth,
    HumanSearchMaskGroundTruth,
)
from aerointentbench.tasks.human_search_segmentation.masks import mask_iou
from aerointentbench.tasks.human_search_segmentation.matching import FrameMatching, match_frame
from aerointentbench.tasks.human_search_segmentation.prediction import MaskPredictedInstance

__all__ = ["HumanSearchSegmentationEvaluator", "QualityScores"]

METRIC_PRECISION: Final = "target_precision"
METRIC_RECALL: Final = "target_recall"
METRIC_F1: Final = "target_f1"

#: How quality was measured, recorded in the evaluation details. Absent means the precomputed
#: scalar path (synthetic profile, or legacy scalar-IoU replay); present means IoU was
#: computed from real prediction and ground-truth masks. Together with the executor id this
#: is what distinguishes synthetic profile, legacy replay, and empirical mask replay in a
#: saved result.
QUALITY_EVAL_EMPIRICAL_MASK: Final = "empirical_mask_iou"


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
    """Turns evidence plus ground truth into a standardised task result.

    Two scoring paths, chosen by the ground-truth form, never by the executor. Interval
    ground truth carries a precomputed match on every instance, so scoring reads it directly
    -- unchanged from before, and byte-identical for the profile suite. Per-frame *mask*
    ground truth carries no match: scoring computes IoU from the prediction and ground-truth
    masks, matches one-to-one per frame, and deduplicates the finds by hidden track ID.
    """

    __slots__ = ("_task_spec",)

    def __init__(self, task_spec: TaskSpec) -> None:
        self._task_spec = task_spec

    def evaluate(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchGroundTruth | HumanSearchMaskGroundTruth,
        contract: Contract,
    ) -> TaskEvaluationResult:
        scores = self.score(evidence, ground_truth)
        details: dict[str, Any] = {
            **scores.to_dict(),
            "processed_frames": evidence.processed_frames,
        }
        if isinstance(ground_truth, HumanSearchMaskGroundTruth):
            details["quality_evaluation"] = QUALITY_EVAL_EMPIRICAL_MASK
        return TaskEvaluationResult.against_contract(
            contract,
            value=scores.value_of(contract.quality_metric),
            details=details,
        )

    def score(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchGroundTruth | HumanSearchMaskGroundTruth,
    ) -> QualityScores:
        """Compute all three metrics. Exposed separately so tests can assert the counts."""
        if isinstance(ground_truth, HumanSearchMaskGroundTruth):
            return self._score_empirical(evidence, ground_truth)
        return self._score_precomputed(evidence, ground_truth)

    def _score_precomputed(
        self, evidence: HumanSearchEvidenceRecord, ground_truth: HumanSearchGroundTruth
    ) -> QualityScores:
        rule = self._task_spec.matching_rule

        matched_targets: set[str] = set()
        predicted_identities: set[str] = set()
        matched_identities: set[str] = set()

        for instance in evidence.instances:
            predicted_identities.add(instance.predicted_target_id)
            ground_truth_track_id = getattr(instance, "ground_truth_track_id", None)
            if ground_truth_track_id is None:
                continue
            if not rule.matches(getattr(instance, "mask_iou", 0.0)):
                # Seen, but not segmented well enough to count as found.
                continue
            matched_targets.add(ground_truth_track_id)
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

    def match(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchMaskGroundTruth,
    ) -> tuple[FrameMatching, ...]:
        """Match every frame that has predictions or scored ground truth, one-to-one.

        Exposed so the matching that produces the metrics is inspectable -- a test, or a
        debug dump, can read the exact IoU that earned each match rather than only the totals.
        Ignored ground-truth regions take no part in matching here; their effect on false
        positives is applied during scoring.
        """
        category = self._task_spec.target_type
        rule = self._task_spec.matching_rule
        predictions_by_frame = self._predictions_by_frame(evidence)

        matchings: list[FrameMatching] = []
        for frame_id in self._frame_ids(predictions_by_frame, ground_truth):
            targets = [
                target
                for target in ground_truth.instances_at(frame_id)
                if not target.ignore and target.category == category
            ]
            matchings.append(
                match_frame(frame_id, predictions_by_frame.get(frame_id, []), targets, rule=rule)
            )
        return tuple(matchings)

    def _score_empirical(
        self, evidence: HumanSearchEvidenceRecord, ground_truth: HumanSearchMaskGroundTruth
    ) -> QualityScores:
        category = self._task_spec.target_type or TARGET_CATEGORY
        rule = self._task_spec.matching_rule
        predictions_by_frame = self._predictions_by_frame(evidence)

        found_tracks: set[str] = set()
        false_positive_predictions = 0
        for frame_id in self._frame_ids(predictions_by_frame, ground_truth):
            predictions = predictions_by_frame.get(frame_id, [])
            instances = ground_truth.instances_at(frame_id)
            scored = [g for g in instances if not g.ignore and g.category == category]
            ignored = [g for g in instances if g.ignore and g.category == category]

            matching = match_frame(frame_id, predictions, scored, rule=rule)
            found_tracks |= matching.matched_track_ids

            matched_ids = {match.prediction_id for match in matching.matches}
            for prediction in predictions:
                if prediction.prediction_id in matched_ids:
                    continue
                # A prediction that fell on an ignore region is neither a find nor a fault:
                # the region was flagged as one the mission is not scored on, so it absorbs
                # the prediction rather than charging it as a false positive.
                if any(rule.matches(mask_iou(prediction.mask, region.mask)) for region in ignored):
                    continue
                false_positive_predictions += 1

        total_targets = len(ground_truth.target_track_ids(category))
        true_positives = len(found_tracks)

        # Target-level counting, per the task specification: a track found in any frame is one
        # found target (deduplicated by hidden track ID); every prediction that matched nothing
        # (and no ignore region) is one false positive; every never-matched target is one miss.
        recall = _ratio(true_positives, total_targets, empty=1.0)
        precision = _ratio(true_positives, true_positives + false_positive_predictions, empty=1.0)
        return QualityScores(
            target_precision=precision,
            target_recall=recall,
            target_f1=_harmonic_mean(precision, recall),
            matched_targets=true_positives,
            total_targets=total_targets,
            predicted_targets=true_positives + false_positive_predictions,
            false_positive_targets=false_positive_predictions,
        )

    def _predictions_by_frame(
        self, evidence: HumanSearchEvidenceRecord
    ) -> dict[int, list[MaskPredictedInstance]]:
        """Group the target-category mask predictions by frame, rejecting the wrong kind.

        A mask ground-truth stream must be paired with an empirical replay set: a precomputed
        instance here carries no mask to compare, and silently scoring it as "found nothing"
        would hide the misconfiguration.
        """
        category = self._task_spec.target_type
        by_frame: dict[int, list[MaskPredictedInstance]] = {}
        for instance in evidence.instances:
            if not isinstance(instance, MaskPredictedInstance):
                raise SchemaValidationError(
                    "empirical mask scoring needs mask predictions, but the evidence holds a "
                    f"{type(instance).__name__}; pair a mask ground-truth stream with an "
                    "empirical replay set, not synthetic or legacy-scalar predictions"
                )
            if instance.category != category:
                continue
            by_frame.setdefault(instance.frame_id, []).append(instance)
        return by_frame

    @staticmethod
    def _frame_ids(
        predictions_by_frame: dict[int, list[MaskPredictedInstance]],
        ground_truth: HumanSearchMaskGroundTruth,
    ) -> list[int]:
        return sorted(set(predictions_by_frame) | {frame.frame_id for frame in ground_truth.frames})


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
