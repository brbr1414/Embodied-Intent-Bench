"""Predicted instances, and the synthetic source that generates them.

Prediction payload
------------------
:class:`PredictedInstance` carries two kinds of field, and the split is the whole point:

- **Policy-visible in aggregate**: ``prediction_id``, ``frame_id``, ``predicted_target_id``,
  ``confidence``. These are what a real system knows about its own output.
- **Hidden**: ``ground_truth_track_id`` and ``mask_iou``. Only the evaluator reads them.

The hidden pair stands in for real mask comparison. A production system would store an
actual mask and the evaluator would compute IoU against ground truth; here the similarity
is precomputed. The evaluator's interface is written against ``mask_iou`` and the task
spec's matching rule, so replacing synthesis with real masks changes what fills the field,
not what reads it.

Synthetic generation
--------------------
Deterministic given ``(seed, frame_id, config_id, track_id)``. Hashing is done with
``hashlib``, not the builtin ``hash``, whose string seed is randomised per process -- using
it would make an episode irreproducible across runs, which the benchmark forbids.

Quality flows from the configuration's ``quality_tier``, so the executor stays unaware of
any of this: a better configuration detects a visible target more often, scores a higher
mask IoU, and produces fewer spurious instances.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from aerointentbench.executor.base import ExecutionRequest
from aerointentbench.schemas.profile import ProfileCatalog, QualityTier
from aerointentbench.tasks.human_search_segmentation.ground_truth import HumanSearchGroundTruth

__all__ = [
    "DEFAULT_TIER_BEHAVIOUR",
    "FramePrediction",
    "PredictedInstance",
    "SyntheticHumanSearchPredictions",
    "TierBehaviour",
]


@dataclass(frozen=True, slots=True)
class PredictedInstance:
    """One predicted person instance."""

    prediction_id: str
    frame_id: int
    #: The system's own identity for this target, as a tracker would assign. Distinct from
    #: the ground-truth track: this is what the system *believes*, and it is the key the
    #: policy-visible unique-target count is computed from.
    predicted_target_id: str
    confidence: float
    #: Hidden. The ground-truth target this instance actually corresponds to, or ``None``
    #: for a false positive. Read only by the evaluator.
    ground_truth_track_id: str | None = None
    #: Hidden. Mask overlap against that ground-truth instance, compared against the task
    #: spec's matching rule. Zero for a false positive.
    mask_iou: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation, for the episode record only -- this includes hidden fields."""
        return {
            "prediction_id": self.prediction_id,
            "frame_id": self.frame_id,
            "predicted_target_id": self.predicted_target_id,
            "confidence": self.confidence,
            "ground_truth_track_id": self.ground_truth_track_id,
            "mask_iou": self.mask_iou,
        }


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """Everything predicted for one frame. This is the ``ExecutionResult.prediction`` payload."""

    frame_id: int
    instances: tuple[PredictedInstance, ...] = ()

    def __len__(self) -> int:
        return len(self.instances)


@dataclass(frozen=True, slots=True)
class TierBehaviour:
    """How a quality tier behaves: how often it sees a target, how well, and how noisily."""

    #: Probability of producing an instance for a target that is genuinely in view.
    detection_probability: float
    #: Mask IoU is drawn uniformly from this range. A range straddling the matching
    #: threshold is what makes a weak configuration produce detections that fail to match --
    #: seeing something is not the same as segmenting it well enough to count.
    mask_iou_range: tuple[float, float]
    confidence_range: tuple[float, float]
    #: Probability per processed frame of inventing an instance that matches nothing.
    false_positive_rate: float


#: Synthetic behaviour per tier. Placeholder values, chosen so that the tiers separate on
#: short-visibility targets rather than on every frame; see docs/v1_spec.md §4.12.
DEFAULT_TIER_BEHAVIOUR: Final[Mapping[QualityTier, TierBehaviour]] = {
    QualityTier.LOW: TierBehaviour(
        detection_probability=0.55,
        mask_iou_range=(0.30, 0.75),
        confidence_range=(0.45, 0.80),
        false_positive_rate=0.004,
    ),
    QualityTier.MEDIUM: TierBehaviour(
        detection_probability=0.75,
        mask_iou_range=(0.42, 0.88),
        confidence_range=(0.60, 0.90),
        false_positive_rate=0.002,
    ),
    QualityTier.HIGH: TierBehaviour(
        detection_probability=0.92,
        mask_iou_range=(0.48, 0.96),
        confidence_range=(0.70, 0.97),
        false_positive_rate=0.001,
    ),
}


def _unit(*parts: object) -> float:
    """Map arbitrary parts to a stable value in [0, 1).

    ``hashlib`` rather than the builtin ``hash``: Python randomises string hashing per
    process, so the builtin would make the same episode score differently on each run.
    """
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / 2**64


def _in_range(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return low + value * (high - low)


class SyntheticHumanSearchPredictions:
    """Generates predictions from hidden ground truth and the configuration's quality tier.

    Plugs into ``ProfileExecutor`` through the ``PredictionSource`` protocol, so the executor
    needs no knowledge of this task. It reads ground truth, which is legitimate -- it is
    standing in for a model that would actually be looking at the scene -- but nothing it
    produces lets a *policy* recover a ground-truth quantity: the hidden fields it writes are
    excluded from the tracker's policy summary.
    """

    __slots__ = ("_behaviour", "_ground_truth", "_profiles")

    def __init__(
        self,
        ground_truth: HumanSearchGroundTruth,
        profiles: ProfileCatalog,
        *,
        behaviour: Mapping[QualityTier, TierBehaviour] | None = None,
    ) -> None:
        self._ground_truth = ground_truth
        self._profiles = profiles
        self._behaviour = dict(behaviour or DEFAULT_TIER_BEHAVIOUR)

    def prediction_for(self, request: ExecutionRequest) -> FramePrediction:
        config_id = request.configuration.config_id
        tier = self._profiles.get(config_id).quality_tier
        behaviour = self._behaviour[tier]

        instances: list[PredictedInstance] = []
        for target in self._ground_truth.visible_at(request.frame_id):
            roll = _unit(request.seed, "detect", request.frame_id, config_id, target.track_id)
            if roll >= behaviour.detection_probability:
                continue
            instances.append(
                PredictedInstance(
                    prediction_id=f"P{request.frame_id}_{target.track_id}",
                    frame_id=request.frame_id,
                    # A tracker re-identifies the same person across frames, so the predicted
                    # identity is stable per target. It is an opaque label, not the ground
                    # truth track ID, and reveals no ground-truth quantity by itself.
                    predicted_target_id=self._predicted_target_id(request.seed, target.track_id),
                    confidence=_in_range(
                        _unit(request.seed, "conf", request.frame_id, config_id, target.track_id),
                        behaviour.confidence_range,
                    ),
                    ground_truth_track_id=target.track_id,
                    mask_iou=_in_range(
                        _unit(request.seed, "iou", request.frame_id, config_id, target.track_id),
                        behaviour.mask_iou_range,
                    ),
                )
            )

        instances.extend(self._false_positives(request, behaviour))
        return FramePrediction(frame_id=request.frame_id, instances=tuple(instances))

    def _false_positives(
        self, request: ExecutionRequest, behaviour: TierBehaviour
    ) -> list[PredictedInstance]:
        config_id = request.configuration.config_id
        if _unit(request.seed, "fp", request.frame_id, config_id) >= behaviour.false_positive_rate:
            return []
        return [
            PredictedInstance(
                prediction_id=f"P{request.frame_id}_FP",
                frame_id=request.frame_id,
                predicted_target_id=f"PT_FP_{request.frame_id}",
                confidence=_in_range(
                    _unit(request.seed, "fpconf", request.frame_id, config_id),
                    behaviour.confidence_range,
                ),
                ground_truth_track_id=None,
                mask_iou=0.0,
            )
        ]

    @staticmethod
    def _predicted_target_id(seed: int, track_id: str) -> str:
        return "PT_" + hashlib.blake2b(f"{seed}|{track_id}".encode(), digest_size=4).hexdigest()
