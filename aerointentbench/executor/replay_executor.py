"""Replay execution: results come from a precomputed record per frame and configuration.

The long-term mapping the benchmark wants is ``frame x config -> prediction``. Computing
that once with real models and replaying it makes every subsequent run reproducible, free of
GPUs, and comparable across policies -- two policies that select the same configuration on
the same frame see exactly the same prediction, so a difference in their scores comes from
their decisions and nothing else.

This module owns the *envelope*: which frame, which configuration, what it cost, and an
opaque prediction payload. The payload's shape belongs to the task, and is passed through
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aerointentbench.executor.base import (
    ExecutionRequest,
    ExecutionResult,
    FailureReason,
    Prediction,
)
from aerointentbench.schemas.loading import DocumentReader, SchemaValidationError, open_document

__all__ = ["ReplayExecutor", "ReplayRecord", "load_replay_records"]

_MS_PER_S: Final = 1000.0
_DOCUMENT_FIELDS: Final = ("record_set_id", "episode_id", "records")
_RECORD_FIELDS: Final = (
    "frame_id",
    "config_id",
    "success",
    "latency_ms",
    "onboard_energy_j",
    "upload_mb",
    "download_mb",
    "failure_reason",
    "prediction",
)


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One precomputed execution outcome."""

    frame_id: int
    config_id: str
    success: bool
    latency_ms: float
    onboard_energy_j: float
    upload_mb: float = 0.0
    download_mb: float = 0.0
    failure_reason: FailureReason | None = None
    prediction: Prediction | None = None

    @property
    def key(self) -> tuple[int, str]:
        return (self.frame_id, self.config_id)


class ReplayExecutor:
    """Serves precomputed records, keyed by frame and configuration."""

    __slots__ = ("_records", "_strict")

    def __init__(
        self, records: Mapping[tuple[int, str], ReplayRecord], *, strict: bool = False
    ) -> None:
        """Args:
        records: Outcomes keyed by ``(frame_id, config_id)``.
        strict: Whether a missing record raises instead of returning a failed result.
            The default is lenient because a record set need not cover every frame of a
            long mission -- a slow configuration skips frames, so a policy can reach a
            frame nothing was precomputed for. Set it when a record set is meant to be
            exhaustive and a gap is a fixture bug.
        """
        self._records = dict(records)
        self._strict = strict

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted({frame_id for frame_id, _ in self._records}))

    def __len__(self) -> int:
        return len(self._records)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        record = self._records.get((request.frame_id, request.configuration.config_id))
        if record is None:
            if self._strict:
                raise SchemaValidationError(
                    f"replay record set has no entry for frame {request.frame_id} and "
                    f"configuration {request.configuration.config_id!r}"
                )
            # Not an exception: a policy is allowed to reach a frame nobody precomputed.
            # It is recorded as a failed inference so the gap is visible in the metrics
            # rather than silently scoring as a success with no evidence.
            return ExecutionResult(
                success=False,
                latency_s=0.0,
                onboard_energy_j=0.0,
                failure_reason=FailureReason.NO_PREDICTION_AVAILABLE,
                metadata={
                    "frame_id": request.frame_id,
                    "config_id": request.configuration.config_id,
                },
            )

        return ExecutionResult(
            success=record.success,
            latency_s=record.latency_ms / _MS_PER_S,
            onboard_energy_j=record.onboard_energy_j,
            upload_mb=record.upload_mb,
            download_mb=record.download_mb,
            prediction=record.prediction if record.success else None,
            failure_reason=record.failure_reason,
            metadata={"source": "replay"},
        )


def load_replay_records(path: Path) -> dict[tuple[int, str], ReplayRecord]:
    """Load and validate a replay record set."""
    reader = open_document(path, document_type="ReplayRecordSet", allowed_fields=_DOCUMENT_FIELDS)
    records: dict[tuple[int, str], ReplayRecord] = {}

    for entry in reader.get_object_list("records", allowed_fields=_RECORD_FIELDS):
        record = _read_record(entry)
        if record.key in records:
            raise SchemaValidationError(
                f"{path.name} -> ReplayRecordSet: duplicate record for frame "
                f"{record.frame_id} and configuration {record.config_id!r}"
            )
        records[record.key] = record

    return records


def _read_record(reader: DocumentReader) -> ReplayRecord:
    success = reader.get_bool("success")
    failure_reason = reader.get_optional_str("failure_reason")

    if success and failure_reason is not None:
        raise SchemaValidationError(f"{reader.context}: a successful record has no failure_reason")
    if not success and failure_reason is None:
        raise SchemaValidationError(
            f"{reader.context}: a failed record must state a failure_reason"
        )

    return ReplayRecord(
        frame_id=reader.get_int("frame_id", minimum=0),
        config_id=reader.get_str("config_id"),
        success=success,
        latency_ms=reader.get_float("latency_ms", minimum=0.0),
        onboard_energy_j=reader.get_float("onboard_energy_j", minimum=0.0),
        upload_mb=reader.get_optional_float("upload_mb", default=0.0, minimum=0.0),
        download_mb=reader.get_optional_float("download_mb", default=0.0, minimum=0.0),
        failure_reason=None if failure_reason is None else _read_failure_reason(reader),
        prediction=reader.get_passthrough("prediction"),
    )


def _read_failure_reason(reader: DocumentReader) -> FailureReason:
    return reader.get_enum("failure_reason", FailureReason)


def make_replay_executor(path: Path, *, strict: bool = False) -> ReplayExecutor:
    """Build a replay executor from a record-set file. Used by the executor registry."""
    return ReplayExecutor(load_replay_records(path), strict=strict)
