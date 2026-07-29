"""Deterministic hard-scenario family generation (V2.4).

One hand-designed hard scenario proves possibility; a scenario *family* is what lets
the benchmark ask whether "adaptive beats static" is a pattern or a coincidence. This
module derives controlled variants of a base hard-trade-off scenario
(``demo_img1_hard_tradeoff.json``) that keep the qualitative structure fixed while
varying the quantities a field mission would not control:

- **Small early targets** (light-model-defeating, ~32 px): which capture slot they
  occupy on the early lanes, lateral offset, exact size within the validated 30-34 px
  band, pose, rotation.
- **Large late targets** (strong-cadence-defeating): which odd capture slot on the
  final lane, lateral offset, size, pose, rotation. Their ~1.5 s visibility windows
  stay centred near odd capture times — that misalignment with the strong executor's
  2-capture cadence *is* the family's defining stress, not a tunable.
- **Battery capacity** within a narrow band, which moves the adaptive policy's
  battery-pressure switch time and probes the fragility of its final margin.

What is deliberately NOT varied: the executor configs (configured latencies/energies),
the contract, the trajectory, the camera, and the world image. Outcome variation must
come from where targets fall and how the real models respond — never from forcing a
winner. Seeds that break the hypothesis are kept and reported.

Determinism: every variant is a pure function of
``(FAMILY_VERSION, base scenario id, seed)``. All sampled values are rounded, recorded
under ``provenance.scenario_family``, and the emitted JSON is strict
(``allow_nan=False``, sorted keys), so identical inputs yield identical bytes. No
wall-clock, no global RNG.

Version history (a sampling change is a version bump — variant identity depends on it):

- **1.0** (never released): sampled late slots from the whole final lane (>= ~37 s)
  with +-0.2 s jitter. A 5-seed smoke run exposed two mismatches with the documented
  hard-scenario structure: late targets could appear *before* the adaptive policy's
  battery-pressure switch (measuring switch-timing luck, not adaptation value), and
  the jitter let wide targets clip the edge of a strong-cadence frame. Finding kept
  in the V2.4 report as sensitivity evidence.
- **1.1**: late slots start at >= 41 s (the post-switch window the hard-scenario
  design defines), late jitter is +-0.1 s, and the late window runs to just before
  path completion. On the base trajectory that fixes the late slot *set* at
  {41, 43, 45, 47}; late-target variation comes from jitter, lateral offset, size,
  pose, rotation, background patch, and battery capacity — not slot choice.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Final

from aerointentbench.schemas.loading import SchemaValidationError, read_json_object
from aerointentbench.v2.scenario import V2Scenario, load_scenario
from aerointentbench.v2.trajectory import PolylineTrajectory, build_trajectory

__all__ = ["FAMILY_VERSION", "variant_id", "write_variant"]

#: Version of the family sampling scheme; bump on any change to what or how we sample.
FAMILY_VERSION: Final = "1.1"

#: Asset aspect ratios (width/height) of the poses the base scenario uses, from the
#: asset manifest's nominal physical dimensions.
_POSES: Final = {
    "walking": {"asset_id": "person_aerial_walking_001", "aspect": 0.755 / 1.5},
    "standing": {"asset_id": "person_aerial_standing_001", "aspect": 1.42 / 2.25},
}
#: Small-target ground extent (m): the Stage-B-validated "light model detects nothing"
#: band (~30-34 px at the base scenario's 42.7 px/m).
_SMALL_HEIGHT_RANGE: Final = (0.70, 0.80)
#: Large late-target extents per pose (m) — comfortably detectable by both models.
_LATE_HEIGHT_RANGE: Final = {"standing": (2.0, 2.25), "walking": (1.40, 1.60)}
_LATERAL_OFFSET_MAX_M: Final = 1.2
_ROTATION_MAX_DEG: Final = 45.0
_BATTERY_CAPACITY_RANGE_WH: Final = (6.60, 7.00)
#: Time jitter around the chosen capture slot. Small targets stay within +-0.3 s of an
#: even slot (their window always covers a strong-processed capture); late targets stay
#: within +-0.1 s of an odd slot (their window never covers one, up to a possible
#: few-pixel sliver of the widest targets at a frame edge — kept as genuine noise).
_SMALL_JITTER_S: Final = 0.3
_LATE_JITTER_S: Final = 0.1
#: Late targets belong to the post-battery-pressure window the hard-scenario design
#: defines; slots earlier than this measured switch-timing luck (family 1.0 finding).
_LATE_WINDOW_START_S: Final = 41.0
#: The final lane may end mid-slot; a capture still happens right up to path completion
#: (the reference scenario's last target sits 0.25 s before the end), so the late
#: window only needs a small end margin.
_LATE_END_MARGIN_S: Final = 0.25
#: Keep sampled times at least this far from a lane's ends (turns distort geometry).
_LANE_MARGIN_S: Final = 1.5
_SMALL_TARGET_COUNT: Final = 4
_LATE_TARGET_COUNT: Final = 4


def variant_id(base_scenario_id: str, seed: int) -> str:
    return f"{base_scenario_id}_SEED_{seed:04d}"


def write_variant(base_scenario_path: Path, seed: int, output_path: Path) -> Path:
    """Generate one deterministic variant, write it, and strictly re-validate it."""
    payload = generate_variant(base_scenario_path, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_scenario(output_path)  # the generator never ships an invalid scenario
    return output_path


def generate_variant(base_scenario_path: Path, seed: int) -> dict[str, Any]:
    """A variant payload as a pure function of (family version, base id, seed)."""
    if seed < 0:
        raise SchemaValidationError(f"seed must be non-negative, got {seed}")
    base_raw = read_json_object(base_scenario_path)
    base = load_scenario(base_scenario_path)
    rng = random.Random(
        f"aerointentbench.v2.scenario_family/{FAMILY_VERSION}/{base.scenario_id}/{seed}"
    )
    trajectory = build_trajectory(base.trajectory, base.drone.speed_mps)

    early_slots, late_slots = _capture_slots(trajectory)
    small_times = sorted(rng.sample(early_slots, _SMALL_TARGET_COUNT))
    late_times = sorted(rng.sample(late_slots, _LATE_TARGET_COUNT))

    late_poses = ["standing", "standing", "walking", "walking"]
    rng.shuffle(late_poses)

    targets: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    for index, slot in enumerate(small_times):
        pose = rng.choice(["walking", "standing"])
        height = round(rng.uniform(*_SMALL_HEIGHT_RANGE), 3)
        targets.append(
            _target(
                object_id=f"TGT_S{index + 1}_{pose.upper()}_SMALL",
                pose=pose,
                height_m=height,
                slot=slot,
                jitter=round(rng.uniform(-_SMALL_JITTER_S, _SMALL_JITTER_S), 2),
                lateral=round(rng.uniform(-_LATERAL_OFFSET_MAX_M, _LATERAL_OFFSET_MAX_M), 3),
                rotation=round(rng.uniform(-_ROTATION_MAX_DEG, _ROTATION_MAX_DEG), 1),
                z_order=10 + index,
                trajectory=trajectory,
                base=base,
                sampled=sampled,
            )
        )
    for index, slot in enumerate(late_times):
        pose = late_poses[index]
        height = round(rng.uniform(*_LATE_HEIGHT_RANGE[pose]), 3)
        targets.append(
            _target(
                object_id=f"TGT_E{index + 1}_{pose.upper()}_LATE",
                pose=pose,
                height_m=height,
                slot=slot,
                jitter=round(rng.uniform(-_LATE_JITTER_S, _LATE_JITTER_S), 2),
                lateral=round(rng.uniform(-_LATERAL_OFFSET_MAX_M, _LATERAL_OFFSET_MAX_M), 3),
                rotation=round(rng.uniform(-_ROTATION_MAX_DEG, _ROTATION_MAX_DEG), 1),
                z_order=20 + index,
                trajectory=trajectory,
                base=base,
                sampled=sampled,
            )
        )

    distractor = _distractor(rng, trajectory, base)
    battery_wh = round(rng.uniform(*_BATTERY_CAPACITY_RANGE_WH), 3)

    payload = dict(base_raw)
    payload["scenario_id"] = variant_id(base.scenario_id, seed)
    payload["random_seed"] = seed
    payload["drone"] = {**base_raw["drone"], "battery_capacity_wh": battery_wh}
    payload["objects"] = [*targets, distractor]
    payload["provenance"] = {
        **dict(base_raw.get("provenance") or {}),
        "scenario_family": {
            "family_version": FAMILY_VERSION,
            "base_scenario_id": base.scenario_id,
            "base_scenario_sha256": hashlib.sha256(base_scenario_path.read_bytes()).hexdigest(),
            "seed": seed,
            "battery_capacity_wh": battery_wh,
            "targets": sampled,
        },
    }
    return payload


# --- geometry helpers ------------------------------------------------------------------------


def _capture_slots(trajectory: PolylineTrajectory) -> tuple[list[int], list[int]]:
    """Usable integer capture times: even slots on the early lanes, odd on the last.

    Lanes are the trajectory's long straight segments; short connecting turns are
    excluded (with a margin) so every sampled position sits on a clean lane run.
    """
    spans = _lane_time_spans(trajectory)
    if len(spans) < 2:
        raise SchemaValidationError(
            "the scenario family needs at least two lanes (early + late); "
            f"found {len(spans)} straight segments"
        )
    *early_spans, late_span = spans
    early = [
        t
        for start, end in early_spans
        for t in range(int(start) + 1, int(end) + 1)
        if t % 2 == 0 and start + _LANE_MARGIN_S <= t <= end - _LANE_MARGIN_S
    ]
    late = [
        t
        for t in range(int(late_span[0]) + 1, int(late_span[1]) + 1)
        if t % 2 == 1
        and t >= _LATE_WINDOW_START_S
        and late_span[0] + _LANE_MARGIN_S <= t <= late_span[1] - _LATE_END_MARGIN_S
    ]
    if len(early) < _SMALL_TARGET_COUNT or len(late) < _LATE_TARGET_COUNT:
        raise SchemaValidationError(
            f"not enough capture slots for the family (early={len(early)}, "
            f"late={len(late)}); the base trajectory is too short"
        )
    return early, late


def _lane_time_spans(trajectory: PolylineTrajectory) -> list[tuple[float, float]]:
    """(start_s, end_s) of each long straight segment, in flight order."""
    import math

    lengths = [
        math.dist(a, b)
        for a, b in zip(trajectory.waypoints_m, trajectory.waypoints_m[1:], strict=False)
    ]
    longest = max(lengths)
    spans: list[tuple[float, float]] = []
    start = 0.0
    for length in lengths:
        end = start + length / trajectory.speed_mps
        if length >= 0.5 * longest:
            spans.append((start, end))
        start = end
    return spans


def _target(
    *,
    object_id: str,
    pose: str,
    height_m: float,
    slot: int,
    jitter: float,
    lateral: float,
    rotation: float,
    z_order: int,
    trajectory: PolylineTrajectory,
    base: V2Scenario,
    sampled: list[dict[str, Any]],
) -> dict[str, Any]:
    time_s = round(slot + jitter, 2)
    on_path = trajectory.position_at(time_s)
    position = (round(on_path[0], 3), round(on_path[1] + lateral, 3))
    _check_in_valid_region(base, object_id, position)
    width_m = round(height_m * _POSES[pose]["aspect"], 3)
    sampled.append(
        {
            "object_id": object_id,
            "pose": pose,
            "capture_slot_s": slot,
            "time_jitter_s": jitter,
            "centre_time_s": time_s,
            "lateral_offset_m": lateral,
            "height_m": height_m,
            "rotation_deg": rotation,
        }
    )
    return {
        "object_id": object_id,
        "class_id": "person",
        "is_target": True,
        "render_mode": "image_asset",
        "asset_id": _POSES[pose]["asset_id"],
        "position_m": list(position),
        "width_m": width_m,
        "height_m": height_m,
        "rotation_deg": rotation,
        "z_order": z_order,
    }


def _distractor(
    rng: random.Random, trajectory: PolylineTrajectory, base: V2Scenario
) -> dict[str, Any]:
    time_s = round(rng.uniform(18.0, 28.0), 2)
    on_path = trajectory.position_at(time_s)
    lateral = round(rng.uniform(-1.0, 1.0), 3)
    position = (round(on_path[0], 3), round(on_path[1] + lateral, 3))
    _check_in_valid_region(base, "DIS_DEBRIS", position)
    return {
        "object_id": "DIS_DEBRIS",
        "class_id": "debris",
        "is_target": False,
        "render_mode": "procedural_marker",
        "position_m": list(position),
        "width_m": 1.4,
        "height_m": 0.9,
        "rotation_deg": round(rng.uniform(-45.0, 45.0), 1),
        "z_order": 5,
        "appearance": {
            "shape": "rect",
            "body_rgb": [146, 95, 100],
            "stripe": False,
            "alpha": 1.0,
        },
    }


def _check_in_valid_region(base: V2Scenario, object_id: str, position: tuple[float, float]) -> None:
    region = base.world.valid_region_m
    if region is None:
        return
    x0, y0, x1, y1 = region
    x, y = position
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        raise SchemaValidationError(
            f"family generator placed {object_id!r} at {position} outside "
            f"valid_region_m {region}; adjust the family's slot/offset bounds"
        )
