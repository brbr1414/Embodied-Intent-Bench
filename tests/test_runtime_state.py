"""The policy-visible observation boundary.

``RuntimeState`` is the entire interface between the simulator and a policy. A field added
here without deliberation could hand a policy ground truth or future knowledge, which
would not produce a wrong number -- it would invalidate every result the benchmark
reports. The field set is therefore pinned, and widening it must be a conscious edit to
this test plus ``docs/v1_spec.md``.
"""

from __future__ import annotations

import dataclasses

import pytest

from aerointentbench.schemas.network_trace import NetworkObservation
from aerointentbench.schemas.runtime_state import EvidenceSummary, RuntimeState

EXPECTED_RUNTIME_STATE_FIELDS = {
    "current_time_s",
    "frame_id",
    "battery_frac",
    "power_mode",
    "network",
    "current_config_id",
    "remaining_deadline_s",
    "cumulative_energy_j",
    "cumulative_communication_mb",
    "path_progress",
    "evidence_summary",
}

#: Evidence a policy may see is derived from its own predictions only. Anything computed
#: against ground truth -- true recall, hidden target counts, match outcomes -- is barred.
EXPECTED_EVIDENCE_SUMMARY_FIELDS = {
    "predicted_unique_targets",
    "processed_frames",
    "mean_prediction_confidence",
}


def _state() -> RuntimeState:
    return RuntimeState(
        current_time_s=25.0,
        frame_id=25,
        battery_frac=0.55,
        power_mode="15W",
        network=NetworkObservation(bandwidth_mbps=8.0, rtt_ms=70.0, packet_loss_frac=0.01),
        current_config_id="CFG_REMOTE_STRONG",
        remaining_deadline_s=35.0,
        cumulative_energy_j=5250.0,
        cumulative_communication_mb=19.5,
        path_progress=0.42,
        evidence_summary=EvidenceSummary(
            predicted_unique_targets=3,
            processed_frames=20,
            mean_prediction_confidence=0.78,
        ),
    )


def test_runtime_state_exposes_exactly_the_permitted_fields() -> None:
    fields = {field.name for field in dataclasses.fields(RuntimeState)}
    assert fields == EXPECTED_RUNTIME_STATE_FIELDS


def test_evidence_summary_exposes_no_ground_truth_derived_field() -> None:
    fields = {field.name for field in dataclasses.fields(EvidenceSummary)}
    assert fields == EXPECTED_EVIDENCE_SUMMARY_FIELDS


@pytest.mark.parametrize("forbidden", ["recall", "ground_truth", "hidden", "target_count", "match"])
def test_no_policy_visible_field_name_suggests_ground_truth(forbidden: str) -> None:
    """A coarse guard against the obvious leak, alongside the exact field-set assertions."""
    names = {field.name for field in dataclasses.fields(RuntimeState)} | {
        field.name for field in dataclasses.fields(EvidenceSummary)
    }
    assert not [name for name in names if forbidden in name]


def test_runtime_state_is_immutable() -> None:
    state = _state()
    with pytest.raises(AttributeError):
        state.battery_frac = 0.1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        state.evidence_summary.processed_frames = 999  # type: ignore[misc]


def test_runtime_state_serialises_for_step_logs() -> None:
    payload = _state().to_dict()
    assert set(payload) == EXPECTED_RUNTIME_STATE_FIELDS
    assert payload["network"] == {
        "bandwidth_mbps": 8.0,
        "rtt_ms": 70.0,
        "packet_loss_frac": 0.01,
    }
    assert payload["evidence_summary"]["predicted_unique_targets"] == 3
    assert payload["current_config_id"] == "CFG_REMOTE_STRONG"


def test_runtime_state_snapshot_is_json_serialisable() -> None:
    import json

    assert json.loads(json.dumps(_state().to_dict()))["frame_id"] == 25


def test_current_config_id_may_be_absent_on_the_first_step() -> None:
    state = dataclasses.replace(_state(), current_config_id=None)
    assert state.to_dict()["current_config_id"] is None
