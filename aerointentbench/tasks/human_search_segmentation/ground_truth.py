"""Hidden ground truth for human search: who was where, and when.

**Never reachable from a policy.** Only the evaluator and the synthetic prediction source
read this, and the prediction source only uses it to decide what a model would plausibly
have seen -- it never puts a ground-truth quantity into anything the tracker will summarise
for a policy.

Visibility is stored as a frame interval rather than a list, because "this person is in
view from frame 40 to frame 100" is both what the fixture means and far easier to read than
sixty frame numbers. The interval is the only thing that makes latency matter: a target
visible for three frames is missed outright by a configuration slow enough to skip them,
which is precisely the cost the benchmark is trying to measure.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aerointentbench.schemas.loading import SchemaValidationError, open_document

__all__ = ["TASK_ID", "HumanSearchGroundTruth", "TargetTrack", "load_human_search_ground_truth"]

#: The one task V1 implements. Declared here rather than repeated as a literal, and matched
#: against ``contract.task_id`` through the task registry.
TASK_ID: Final = "HUMAN_SEARCH_SEGMENTATION"

_DOCUMENT_FIELDS: Final = ("ground_truth_id", "frame_stream_id", "task_id", "targets")
_TARGET_FIELDS: Final = ("track_id", "first_frame_id", "last_frame_id", "note")


@dataclass(frozen=True, slots=True)
class TargetTrack:
    """One person, visible over an inclusive frame interval."""

    track_id: str
    first_frame_id: int
    last_frame_id: int
    note: str = ""

    def is_visible_at(self, frame_id: int) -> bool:
        return self.first_frame_id <= frame_id <= self.last_frame_id

    @property
    def visible_frame_count(self) -> int:
        return self.last_frame_id - self.first_frame_id + 1


@dataclass(frozen=True, slots=True)
class HumanSearchGroundTruth:
    """Every target that appears in one frame stream.

    Keyed on the stream rather than the episode because ground truth describes the *scene*.
    Several episodes may fly the same stream under different battery, network, or contract
    conditions, and they must all be scored against the same answers.
    """

    frame_stream_id: str
    targets: tuple[TargetTrack, ...]
    ground_truth_id: str = ""

    @property
    def task_id(self) -> str:
        return TASK_ID

    @property
    def track_ids(self) -> frozenset[str]:
        return frozenset(target.track_id for target in self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def visible_at(self, frame_id: int) -> Iterator[TargetTrack]:
        """Yield the targets in view at ``frame_id``, in declaration order."""
        return (target for target in self.targets if target.is_visible_at(frame_id))


def load_human_search_ground_truth(path: Path) -> HumanSearchGroundTruth:
    """Load and validate a ground-truth file."""
    reader = open_document(
        path, document_type="HumanSearchGroundTruth", allowed_fields=_DOCUMENT_FIELDS
    )

    declared_task_id = reader.get_str("task_id")
    if declared_task_id != TASK_ID:
        raise SchemaValidationError(
            f"{reader.context}: this loader reads {TASK_ID!r} ground truth, "
            f"but the file declares {declared_task_id!r}"
        )

    targets: list[TargetTrack] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("targets", allowed_fields=_TARGET_FIELDS):
        first = entry.get_int("first_frame_id", minimum=0)
        last = entry.get_int("last_frame_id", minimum=first)
        track_id = entry.get_str("track_id")
        if track_id in seen:
            raise SchemaValidationError(
                f"{reader.context}: duplicate target track_id {track_id!r}; "
                "deduplication is keyed on it, so it must be unique"
            )
        seen.add(track_id)
        targets.append(
            TargetTrack(
                track_id=track_id,
                first_frame_id=first,
                last_frame_id=last,
                note=entry.get_optional_str("note") or "",
            )
        )

    return HumanSearchGroundTruth(
        frame_stream_id=reader.get_str("frame_stream_id"),
        targets=tuple(targets),
        ground_truth_id=reader.get_optional_str("ground_truth_id") or "",
    )
