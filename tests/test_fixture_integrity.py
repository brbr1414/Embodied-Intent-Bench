"""Every shipped fixture loads, and cross-file references resolve.

Loaders validate one document at a time; nothing in a single file can catch an episode
naming a network trace that does not exist. These tests close that gap, so a fixture set
cannot drift into referencing something absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerointentbench import SCHEMA_VERSION
from aerointentbench.schemas import (
    load_config_catalog,
    load_contract,
    load_episode,
    load_network_trace,
    load_path_spec,
    load_platform_profile,
    load_task_spec,
)

LOADERS_BY_DIRECTORY = {
    "contracts": load_contract,
    "episodes": load_episode,
    "configs": load_config_catalog,
    "platforms": load_platform_profile,
    "task_specs": load_task_spec,
    "network_traces": load_network_trace,
    "paths": load_path_spec,
}


def _fixture_paths(data_dir: Path, directory: str) -> list[Path]:
    return sorted((data_dir / directory).glob("*.json"))


@pytest.mark.parametrize("directory", sorted(LOADERS_BY_DIRECTORY))
def test_every_fixture_in_the_directory_loads(data_dir: Path, directory: str) -> None:
    paths = _fixture_paths(data_dir, directory)
    assert paths, f"expected at least one fixture in data/{directory}"
    for path in paths:
        LOADERS_BY_DIRECTORY[directory](path)


@pytest.mark.parametrize("directory", sorted(LOADERS_BY_DIRECTORY))
def test_every_fixture_declares_the_supported_schema_version(
    data_dir: Path, directory: str
) -> None:
    for path in _fixture_paths(data_dir, directory):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("schema_version") == SCHEMA_VERSION, path


def test_the_three_required_network_conditions_are_present(data_dir: Path) -> None:
    trace_ids = {
        load_network_trace(path).trace_id for path in _fixture_paths(data_dir, "network_traces")
    }
    assert trace_ids == {
        "NETWORK_STABLE_001",
        "NETWORK_DEGRADING_001",
        "NETWORK_DISCONNECTING_001",
    }


def test_episode_references_resolve(data_dir: Path) -> None:
    platform_ids = {
        load_platform_profile(path).platform_id for path in _fixture_paths(data_dir, "platforms")
    }
    trace_ids = {
        load_network_trace(path).trace_id for path in _fixture_paths(data_dir, "network_traces")
    }
    path_ids = {load_path_spec(path).path_id for path in _fixture_paths(data_dir, "paths")}
    known_config_ids: set[str] = set()
    for path in _fixture_paths(data_dir, "configs"):
        known_config_ids.update(load_config_catalog(path).ids())

    for path in _fixture_paths(data_dir, "episodes"):
        episode = load_episode(path)
        assert episode.platform_id in platform_ids, path
        assert episode.network_trace_id in trace_ids, path
        assert episode.path_id in path_ids, path
        assert set(episode.allowed_config_ids) <= known_config_ids, path


def test_contract_references_resolve(data_dir: Path) -> None:
    specs = {
        spec.task_id: spec
        for spec in (load_task_spec(path) for path in _fixture_paths(data_dir, "task_specs"))
    }
    for path in _fixture_paths(data_dir, "contracts"):
        contract = load_contract(path)
        assert contract.task_id in specs, path
        spec = specs[contract.task_id]
        assert spec.supports_metric(contract.quality_metric), path
        assert contract.evidence_type == spec.evidence_type, path


def test_every_network_trace_covers_the_longest_contract_deadline(data_dir: Path) -> None:
    """A trace that ends before the deadline would make late observations undefined."""
    longest_deadline = max(
        load_contract(path).deadline_s for path in _fixture_paths(data_dir, "contracts")
    )
    for path in _fixture_paths(data_dir, "network_traces"):
        trace = load_network_trace(path)
        assert trace.start_s == 0.0, path
        assert trace.end_s >= longest_deadline, path


def test_measurement_bearing_fixtures_are_marked_synthetic(data_dir: Path) -> None:
    """Profiles, traces, predictions, and ground truth carry invented numbers.

    Their filenames say so, so that a value lifted out of this repository cannot be
    mistaken for a measurement. Specification files (contracts, episodes, task specs,
    catalogs) state choices rather than measurements and are exempt.
    """
    for directory in ("platforms", "network_traces"):
        for path in _fixture_paths(data_dir, directory):
            assert path.name.startswith("synthetic_"), (
                f"{path} carries invented numbers and must be named synthetic_*"
            )
