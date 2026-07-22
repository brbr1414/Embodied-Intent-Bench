"""The reference suite: what every baseline scores on the shipped fixtures, pinned.

This is the regression test for the benchmark *as a benchmark*. Every other test checks that
a component behaves; this one checks that the numbers the benchmark reports have not moved.
Any change to a fixture, a profile, the synthetic prediction model, the evaluator, or a
policy shows up here as a diff in ``tests/reference/baseline_results.json`` -- which is the
point. Numbers that change silently are numbers nobody can compare across versions.

Regenerating is deliberate, not automatic::

    python -m tests.test_reference_suite --update

A regenerated file must be reviewed like any other change. If the diff is not one you
intended, the benchmark just moved under you.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aerointentbench.benchmark import BenchmarkData, run_suite
from aerointentbench.schemas.contract import load_contract

REFERENCE_PATH = Path(__file__).parent / "reference" / "baseline_results.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every baseline, plus the profile-blind variant. Ordered so the file reads consistently.
REFERENCE_RUNS: tuple[tuple[str, str, bool], ...] = (
    ("always_local_light", "contract_001", True),
    ("always_local_strong", "contract_001", True),
    ("always_remote_strong", "contract_001", True),
    ("rule_based", "contract_001", True),
    ("rule_based_profiles_hidden", "contract_001", False),
    ("always_local_strong", "contract_002_local_only", True),
    ("rule_based", "contract_002_local_only", True),
)

#: Rounded before comparison. Floating-point arithmetic can differ in the last bits across
#: platforms, and a benchmark result that changed only there has not actually changed.
_PLACES = 6


def _digest(policy_name: str, contract_name: str, *, disclose: bool) -> dict[str, Any]:
    """Run one policy over the whole suite and reduce it to a comparable summary.

    Deliberately not the full result file: that runs to megabytes and embeds ground truth.
    This keeps the figures a reader would actually compare between two versions.
    """
    data = BenchmarkData(REPO_ROOT / "data")
    contract = load_contract(REPO_ROOT / "data" / "contracts" / f"{contract_name}.json")
    result = run_suite(
        data=data,
        episodes=data.episodes(),
        contract=contract,
        policy_name=policy_name.removesuffix("_profiles_hidden"),
        disclose_profiles=disclose,
    )
    aggregate = result.aggregate
    return {
        "mission_success_rate": round(aggregate.mission_success_rate, _PLACES),
        "quality_success_rate": round(aggregate.quality_success_rate, _PLACES),
        "communication_constraint_success_rate": round(
            aggregate.communication_constraint_success_rate, _PLACES
        ),
        "battery_constraint_success_rate": round(
            aggregate.battery_constraint_success_rate, _PLACES
        ),
        "privacy_constraint_success_rate": round(
            aggregate.privacy_constraint_success_rate, _PLACES
        ),
        "mean_quality_value": round(aggregate.mean_quality_value, _PLACES),
        "episodes": [
            {
                "episode_id": episode.metrics.episode_id,
                "mission_success": episode.metrics.mission_success,
                "quality_value": round(episode.metrics.quality_value, _PLACES),
                "final_battery_fraction": round(episode.metrics.final_battery_fraction, _PLACES),
                "total_communication_mb": round(episode.metrics.total_communication_mb, _PLACES),
                "mission_completion_time_s": round(
                    episode.metrics.mission_completion_time_s, _PLACES
                ),
                "mean_end_to_end_inference_latency_ms": round(
                    episode.metrics.mean_end_to_end_inference_latency_ms, _PLACES
                ),
                "configuration_switch_count": episode.metrics.configuration_switch_count,
                "constraint_violation_count": episode.metrics.constraint_violation_count,
                "invalid_action_count": episode.metrics.invalid_action_count,
                "failed_inference_count": episode.metrics.failed_inference_count,
                "processed_frames": episode.metrics.processed_frames,
                "termination_reason": episode.metrics.termination_reason,
            }
            for episode in result.episodes
        ],
    }


def build_reference() -> dict[str, Any]:
    return {
        f"{policy}@{contract}": _digest(policy, contract, disclose=disclose)
        for policy, contract, disclose in REFERENCE_RUNS
    }


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    if not REFERENCE_PATH.is_file():
        pytest.fail(
            f"missing reference results at {REFERENCE_PATH}; "
            "regenerate with `python -m tests.test_reference_suite --update`"
        )
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual() -> dict[str, Any]:
    return build_reference()


def test_the_reference_file_covers_every_run(reference) -> None:
    assert set(reference) == {f"{policy}@{contract}" for policy, contract, _ in REFERENCE_RUNS}


@pytest.mark.parametrize("key", [f"{policy}@{contract}" for policy, contract, _ in REFERENCE_RUNS])
def test_results_match_the_reference(reference, actual, key: str) -> None:
    """A benchmark whose numbers move silently cannot be compared across versions."""
    assert actual[key] == reference[key], (
        f"{key} no longer matches the reference. If the change is intended, regenerate with "
        "`python -m tests.test_reference_suite --update` and review the diff."
    )


# --- properties the reference itself must have -------------------------------------------


def test_adaptation_still_beats_every_static_baseline(actual) -> None:
    """The benchmark's reason to exist, pinned as a property rather than a number."""
    adaptive = actual["rule_based@contract_001"]["mission_success_rate"]
    statics = [
        actual[f"{name}@contract_001"]["mission_success_rate"]
        for name in ("always_local_light", "always_local_strong", "always_remote_strong")
    ]
    assert adaptive > max(statics)


def test_the_baselines_still_fail_for_different_reasons(actual) -> None:
    """A suite where every baseline fails the same way would not be informative."""
    light = actual["always_local_light@contract_001"]
    remote = actual["always_remote_strong@contract_001"]

    assert light["quality_success_rate"] == 0.0
    assert light["communication_constraint_success_rate"] == 1.0
    assert remote["quality_success_rate"] == 1.0
    assert remote["communication_constraint_success_rate"] == 0.0


def test_hiding_profiles_still_costs_the_policy(actual) -> None:
    """The measured justification for disclosing an ordinal quality tier."""
    seen = actual["rule_based@contract_001"]["mission_success_rate"]
    blind = actual["rule_based_profiles_hidden@contract_001"]["mission_success_rate"]
    assert seen > blind


def test_no_shipped_episode_is_unpassable(actual) -> None:
    """An episode nothing can pass measures nothing.

    Checked across every baseline: each shipped episode must be passed by at least one of
    them under the default contract.
    """
    runs = [value for key, value in actual.items() if key.endswith("@contract_001")]
    passed_by_someone: dict[str, bool] = {}
    for run in runs:
        for episode in run["episodes"]:
            passed_by_someone.setdefault(episode["episode_id"], False)
            passed_by_someone[episode["episode_id"]] |= episode["mission_success"]

    unpassable = sorted(name for name, passed in passed_by_someone.items() if not passed)
    assert not unpassable, f"no baseline can pass {unpassable}"


def test_the_privacy_contract_blocks_remote_execution(actual) -> None:
    """Under local_only, remote is never executed -- so nothing is transmitted."""
    for episode in actual["rule_based@contract_002_local_only"]["episodes"]:
        assert episode["total_communication_mb"] == 0.0


def _main() -> int:
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(build_reference(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
