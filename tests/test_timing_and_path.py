"""Decision-loop timing and path progress.

These rules decide how many frames an expensive configuration costs the mission, which is
most of what the benchmark measures, so the boundaries are pinned explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aerointentbench.schemas.path import load_path_spec
from aerointentbench.simulator.path import ConstantVelocityPath
from aerointentbench.simulator.timing import (
    elapsed_step_time_s,
    frame_id_at,
    frames_skipped,
)

# --- step duration ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latency_s", "expected"),
    [
        (0.10, 1.0),  # fast inference does not buy a second decision
        (0.45, 1.0),
        (1.00, 1.0),  # exactly the interval
        (1.01, 1.01),  # one inference at a time: the step stretches
        (2.50, 2.50),
        (0.00, 1.0),
    ],
)
def test_step_lasts_the_longer_of_interval_and_latency(latency_s: float, expected: float) -> None:
    assert elapsed_step_time_s(latency_s) == expected


def test_decision_interval_is_configurable() -> None:
    assert elapsed_step_time_s(0.2, decision_interval_s=0.5) == 0.5
    assert elapsed_step_time_s(0.8, decision_interval_s=0.5) == 0.8


# --- frame indexing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_s", "expected_frame"),
    [(0.0, 0), (0.9, 0), (1.0, 1), (1.9, 1), (2.0, 2), (25.0, 25), (60.0, 60)],
)
def test_frame_index_floors_the_clock(time_s: float, expected_frame: int) -> None:
    """At t=1.9 s the frame captured at 2.0 s does not exist yet."""
    assert frame_id_at(time_s) == expected_frame


def test_a_slow_inference_skips_the_frames_it_ran_past() -> None:
    start = frame_id_at(10.0)
    after_fast = frame_id_at(10.0 + elapsed_step_time_s(0.4))
    after_slow = frame_id_at(10.0 + elapsed_step_time_s(3.2))

    assert frames_skipped(start, after_fast) == 0
    assert after_slow - start == 3
    assert frames_skipped(start, after_slow) == 2


def test_frames_skipped_never_reports_negative() -> None:
    assert frames_skipped(10, 10) == 0
    assert frames_skipped(10, 9) == 0


# --- path progress ------------------------------------------------------------------


@pytest.fixture
def path() -> ConstantVelocityPath:
    """Behaviour-test scale: 250 m at 5 m/s => 50 s, checkable by eye.

    Deliberately not the shipped 4500 m fixture. These tests are about the progress
    *rule*, not the mission's size, and should not have to be rewritten when the
    benchmark is rescaled. The shipped path's own dimensions are asserted in
    test_fixture_integrity.py.
    """
    return ConstantVelocityPath(length_m=250.0, velocity_mps=5.0)


def test_path_duration_follows_length_and_velocity(path: ConstantVelocityPath) -> None:
    assert path.length_m == 250.0
    assert path.duration_s == 50.0


def test_from_spec_takes_its_length_from_the_specification(data_dir: Path) -> None:
    spec = load_path_spec(data_dir / "paths" / "path_001.json")
    built = ConstantVelocityPath.from_spec(spec, velocity_mps=5.0)
    assert built.length_m == spec.length_m
    assert built.duration_s == spec.length_m / 5.0


@pytest.mark.parametrize(
    ("elapsed_s", "expected_progress"),
    [(0.0, 0.0), (12.5, 0.25), (25.0, 0.5), (50.0, 1.0)],
)
def test_progress_is_linear_in_time(
    path: ConstantVelocityPath, elapsed_s: float, expected_progress: float
) -> None:
    assert path.progress_at(elapsed_s) == pytest.approx(expected_progress)


def test_progress_clamps_at_one(path: ConstantVelocityPath) -> None:
    """path_progress must keep the [0, 1] meaning its schema promises."""
    assert path.progress_at(60.0) == 1.0
    assert path.progress_at(10_000.0) == 1.0
    assert path.distance_at(10_000.0) == 250.0


def test_progress_clamps_at_zero(path: ConstantVelocityPath) -> None:
    assert path.progress_at(-5.0) == 0.0


def test_the_vehicle_flies_on_regardless_of_inference_cost(path: ConstantVelocityPath) -> None:
    """A slow configuration does not pause the aircraft; it covers ground unprocessed."""
    slow_step = elapsed_step_time_s(4.0)
    assert path.progress_at(slow_step) == pytest.approx(4 * path.progress_at(1.0))
