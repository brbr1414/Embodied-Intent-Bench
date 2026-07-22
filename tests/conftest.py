"""Shared test fixtures.

Tests read the real files under ``data/`` rather than inventing parallel fixtures, so a
fixture that drifts out of conformance fails the suite instead of quietly diverging from
what the benchmark actually ships.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The repository's benchmark data directory."""
    assert DATA_DIR.is_dir(), f"expected benchmark data at {DATA_DIR}"
    return DATA_DIR


@pytest.fixture
def write_json(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Return a helper that writes a payload to a temporary JSON file and returns its path.

    Used to exercise malformed documents without keeping invalid files in ``data/``.
    """
    counter = 0

    def _write(payload: Any, *, name: str | None = None) -> Path:
        nonlocal counter
        counter += 1
        path = tmp_path / (name or f"document_{counter}.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def valid_contract_payload() -> dict[str, Any]:
    """A minimal conforming contract document, for tests that mutate one field at a time."""
    return {
        "schema_version": "1.0",
        "contract_id": "CONTRACT_TEST",
        "task_id": "HUMAN_SEARCH_SEGMENTATION",
        "evidence_type": "instance_mask_set",
        "quality_metric": "target_f1",
        "quality_operator": ">=",
        "quality_threshold": 0.80,
        "deadline_s": 60.0,
        "communication_budget_mb": 50.0,
        "min_final_battery_frac": 0.20,
        "privacy_level": "remote_allowed",
    }


@pytest.fixture
def valid_trace_payload() -> dict[str, Any]:
    """A minimal conforming two-segment network trace document."""
    return {
        "schema_version": "1.0",
        "trace_id": "NETWORK_TEST",
        "segments": [
            {
                "start_s": 0,
                "end_s": 10,
                "bandwidth_mbps": 20.0,
                "rtt_ms": 30.0,
                "packet_loss_frac": 0.0,
            },
            {
                "start_s": 10,
                "end_s": 20,
                "bandwidth_mbps": 5.0,
                "rtt_ms": 90.0,
                "packet_loss_frac": 0.02,
            },
        ],
    }
