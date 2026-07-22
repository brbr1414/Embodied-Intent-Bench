"""The composition root: turn a data directory plus a policy name into results.

This is the one module allowed to know about every layer at once. It resolves fixtures by
ID, constructs concrete components from the registries, injects them into an
``EpisodeRunner``, and scores what comes back. Everything below it talks through protocols.

Keeping composition here is what makes the extensibility claims real. Swapping the executor,
the battery model, or the task is a change to *this* file or a registry entry -- never to the
runner, the metrics, or a policy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from aerointentbench import SCHEMA_VERSION, __version__
from aerointentbench.executor.base import Executor
from aerointentbench.executor.profile_executor import ProfileExecutor
from aerointentbench.metrics.aggregate_metrics import AggregateMetrics, aggregate_metrics
from aerointentbench.metrics.episode_metrics import EpisodeMetrics, compute_episode_metrics
from aerointentbench.policies.base import Policy
from aerointentbench.policies.registry import policy_registry
from aerointentbench.schemas.configuration import ConfigCatalog, load_config_catalog
from aerointentbench.schemas.contract import Contract, load_contract
from aerointentbench.schemas.episode import Episode, load_episode
from aerointentbench.schemas.loading import SchemaValidationError
from aerointentbench.schemas.network_trace import NetworkTrace, load_network_trace
from aerointentbench.schemas.path import PathSpec, load_path_spec
from aerointentbench.schemas.platform import PlatformProfile, load_platform_profile
from aerointentbench.schemas.profile import (
    ProfileCatalog,
    PublicProfileView,
    check_catalog_is_profiled,
    load_profile_catalog,
)
from aerointentbench.schemas.task_spec import TaskSpec, check_contract_is_supported, load_task_spec
from aerointentbench.simulator.action_validator import (
    DEFAULT_SAFE_FALLBACK_CONFIG_ID,
    ActionValidator,
)
from aerointentbench.simulator.battery_model import BatteryModel, SimpleBatteryModel
from aerointentbench.simulator.episode_runner import EpisodeRunner
from aerointentbench.simulator.network_trace import NetworkModel, TraceBasedNetworkModel
from aerointentbench.simulator.path import ConstantVelocityPath
from aerointentbench.simulator.records import EpisodeRecord
from aerointentbench.simulator.state_manager import StateManager
from aerointentbench.tasks.registry import resolve_task

__all__ = ["BenchmarkData", "EpisodeResult", "SuiteResult", "run_episode", "run_suite"]

_LOGGER: Final = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One scored episode: the full record, and the metrics derived from it."""

    record: EpisodeRecord
    metrics: EpisodeMetrics


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """A policy's results over a set of episodes."""

    policy_name: str
    executor_name: str
    episodes: tuple[EpisodeResult, ...]
    aggregate: AggregateMetrics

    def to_dict(self, *, include_detail: bool = False) -> dict[str, Any]:
        """Serialise to the V1 result-file shape.

        ``include_detail`` adds the step log and the raw evidence. Both are off by default:
        raw evidence carries ground-truth-derived fields, so a shared result file would
        otherwise disclose the answers.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": __version__,
            "policy": self.policy_name,
            "executor": self.executor_name,
            "aggregate": self.aggregate.to_dict(),
            "episodes": [
                {
                    **result.metrics.to_dict(),
                    "record": result.record.to_dict(include_detail=include_detail),
                }
                for result in self.episodes
            ],
        }


class BenchmarkData:
    """Every fixture under a data root, indexed by the IDs that reference it."""

    __slots__ = (
        "_catalog",
        "_ground_truth_dir",
        "_paths",
        "_platforms",
        "_profiles",
        "_root",
        "_task_specs",
        "_traces",
    )

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise SchemaValidationError(f"benchmark data directory not found: {root}")
        self._root = root
        self._catalog = _single_catalog(root / "configs")
        self._platforms = {
            profile.platform_id: profile
            for profile in _load_all(root / "platforms", load_platform_profile)
        }
        self._profiles = {
            catalog.platform_id: catalog
            for catalog in _load_all(root / "profiles", load_profile_catalog)
        }
        self._paths = {spec.path_id: spec for spec in _load_all(root / "paths", load_path_spec)}
        self._traces = {
            trace.trace_id: trace
            for trace in _load_all(root / "network_traces", load_network_trace)
        }
        self._task_specs = {
            spec.task_id: spec for spec in _load_all(root / "task_specs", load_task_spec)
        }
        self._ground_truth_dir = root / "ground_truth"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def catalog(self) -> ConfigCatalog:
        return self._catalog

    @property
    def ground_truth_dir(self) -> Path:
        return self._ground_truth_dir

    def platform(self, platform_id: str) -> PlatformProfile:
        return _lookup(self._platforms, platform_id, "platform")

    def profiles(self, platform_id: str) -> ProfileCatalog:
        return _lookup(self._profiles, platform_id, "configuration profile set for platform")

    def path(self, path_id: str) -> PathSpec:
        return _lookup(self._paths, path_id, "path")

    def trace(self, trace_id: str) -> NetworkTrace:
        return _lookup(self._traces, trace_id, "network trace")

    def task_spec(self, task_id: str) -> TaskSpec:
        return _lookup(self._task_specs, task_id, "task specification")

    def episodes(self) -> tuple[Episode, ...]:
        return tuple(_load_all(self._root / "episodes", load_episode))

    def contracts(self) -> tuple[Contract, ...]:
        return tuple(_load_all(self._root / "contracts", load_contract))


def run_episode(
    *,
    data: BenchmarkData,
    episode: Episode,
    contract: Contract,
    policy_name: str = "rule_based",
    policy: Policy | None = None,
    executor: Executor | None = None,
    executor_name: str = "profile",
    battery_model: BatteryModel | None = None,
    network_model: NetworkModel | None = None,
    disclose_profiles: bool = True,
    fallback_config_id: str = DEFAULT_SAFE_FALLBACK_CONFIG_ID,
) -> EpisodeResult:
    """Assemble the components for one episode, run it, and score the result.

    Args:
        disclose_profiles: Whether policies may consult public configuration profiles. An
            explicit benchmark-mode setting, not something a policy can arrange for itself.
        policy: A constructed policy, overriding ``policy_name``. Lets an external policy be
            evaluated without registering it.
        executor: A constructed executor, overriding the default profile-driven one.
        battery_model: Overrides ``SimpleBatteryModel``. The seam a recorded-discharge or
            electrochemical model plugs into.
        network_model: Overrides the trace-based model. The seam an external simulator
            adapter plugs into, since it is where "what the link is doing" comes from.
    """
    platform = data.platform(episode.platform_id)
    profiles = data.profiles(episode.platform_id)
    task_spec = data.task_spec(contract.task_id)

    # Cross-document checks, up front. A mismatch discovered mid-episode would surface as a
    # confusing failure on whichever step first touched it.
    check_contract_is_supported(
        task_spec, task_id=contract.task_id, quality_metric=contract.quality_metric
    )
    check_catalog_is_profiled(
        profiles, config_ids=episode.allowed_config_ids, platform_id=episode.platform_id
    )

    task = resolve_task(task_spec)
    ground_truth = task.load_ground_truth(data.ground_truth_dir, episode)
    if ground_truth is None:
        _LOGGER.warning(
            "no ground truth for frame stream %s; quality cannot be scored",
            episode.frame_stream_id,
        )

    resolved_policy = (
        policy
        if policy is not None
        else _build_policy(policy_name, profiles, disclose_profiles=disclose_profiles)
    )
    resolved_executor = (
        executor
        if executor is not None
        else ProfileExecutor(
            profiles, predictions=task.create_prediction_source(ground_truth, profiles)
        )
    )

    runner = EpisodeRunner(
        episode=episode,
        contract=contract,
        catalog=data.catalog,
        state_manager=StateManager(
            episode=episode,
            platform=platform,
            path=ConstantVelocityPath.from_spec(
                data.path(episode.path_id), velocity_mps=episode.velocity_mps
            ),
            battery_model=battery_model if battery_model is not None else SimpleBatteryModel(),
        ),
        network_model=(
            network_model
            if network_model is not None
            else TraceBasedNetworkModel(data.trace(episode.network_trace_id))
        ),
        executor=resolved_executor,
        policy=resolved_policy,
        evidence_tracker=task.create_tracker(),
        action_validator=ActionValidator(
            catalog=data.catalog,
            allowed_config_ids=episode.allowed_config_ids,
            privacy_level=contract.privacy_level,
            fallback_config_id=fallback_config_id,
        ),
        policy_name=policy_name,
        executor_name=executor_name,
    )

    record = runner.run()
    evaluation = task.create_evaluator().evaluate(record.evidence, ground_truth, contract)
    return EpisodeResult(
        record=record, metrics=compute_episode_metrics(record, evaluation, contract)
    )


def run_suite(
    *,
    data: BenchmarkData,
    episodes: Sequence[Episode],
    contract: Contract,
    policy_name: str = "rule_based",
    policy: Policy | None = None,
    disclose_profiles: bool = True,
) -> SuiteResult:
    """Run every episode against one contract and aggregate the results."""
    results = tuple(
        run_episode(
            data=data,
            episode=episode,
            contract=contract,
            policy_name=policy_name,
            policy=policy,
            disclose_profiles=disclose_profiles,
        )
        for episode in episodes
    )
    return SuiteResult(
        policy_name=policy_name,
        executor_name="profile",
        episodes=results,
        aggregate=aggregate_metrics([result.metrics for result in results]),
    )


def _build_policy(policy_name: str, profiles: ProfileCatalog, *, disclose_profiles: bool) -> Policy:
    view = profiles.public_view() if disclose_profiles else PublicProfileView.hidden()
    try:
        return policy_registry.create(policy_name, public_profiles=view)
    except TypeError:
        # A policy that takes no profiles is legitimate -- the static baselines do not.
        return policy_registry.create(policy_name)


def _load_all(directory: Path, loader):
    if not directory.is_dir():
        return []
    return [loader(path) for path in sorted(directory.glob("*.json"))]


def _single_catalog(directory: Path) -> ConfigCatalog:
    catalogs = _load_all(directory, load_config_catalog)
    if len(catalogs) != 1:
        raise SchemaValidationError(
            f"expected exactly one configuration catalog in {directory}, found {len(catalogs)}. "
            "V1 runs one catalog per data root; selecting among several is not yet specified."
        )
    return catalogs[0]


def _lookup(index: dict[str, Any], key: str, kind: str):
    try:
        return index[key]
    except KeyError:
        raise SchemaValidationError(
            f"no {kind} {key!r} in the benchmark data; available: {sorted(index)}"
        ) from None


def episodes_by_id(data: BenchmarkData, episode_ids: Iterable[str]) -> tuple[Episode, ...]:
    """Select episodes by ID, preserving the requested order."""
    index = {episode.episode_id: episode for episode in data.episodes()}
    return tuple(_lookup(index, episode_id, "episode") for episode_id in episode_ids)
