"""The V2 scenario schema: one document describing a visual mission end to end.

A scenario names the aerial world, the UAV and its predefined trajectory, the camera,
the synthetic object layer, the executor configurations a policy may select among, and
the mission contract. It is a **separate** schema from the frozen V1 documents -- the V1
contract/episode schemas are not modified to carry visual-simulation fields -- but it
reuses the same strict-validation conventions: a declared version, unknown fields
rejected, and actionable errors.

Coordinate convention (see ``docs/v2_design.md``): the local origin is the top-left
corner of the aerial raster; x grows right, y grows down; world units are meters, image
units are pixels; field names carry unit suffixes (``position_m``, ``speed_mps``,
``width_px``, ``meters_per_pixel``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.contract import Contract, PrivacyLevel
from aerointentbench.schemas.loading import (
    DocumentReader,
    SchemaValidationError,
    SchemaVersionError,
    read_json_object,
)
from aerointentbench.v2.network import NetworkRegime

__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "CameraSpec",
    "DroneSpec",
    "ExecutorConfigSpec",
    "ObjectSpec",
    "SimulationSpec",
    "TrajectorySpec",
    "V2Scenario",
    "WorldSpec",
    "load_scenario",
]

#: The one V2 scenario version this release reads. Any other version fails loudly.
SCENARIO_SCHEMA_VERSION: Final = "2.0"

_TOP_FIELDS: Final = (
    "scenario_schema_version",
    "scenario_id",
    "random_seed",
    "evaluation_purpose",
    "assets_manifest",
    "world",
    "drone",
    "trajectory",
    "camera",
    "simulation",
    "objects",
    "executor_configs",
    "mission_contract",
    "provenance",
)

#: What a scenario's results are allowed to claim. ``controlled_observability`` and
#: ``integration_diagnostic`` results must never be presented as aerial-human
#: perception; only a scenario with genuinely appropriate licensed aerial assets earns
#: ``aerial_target_evaluation``.
_EVALUATION_PURPOSES: Final = (
    "integration_diagnostic",
    "controlled_observability",
    "aerial_target_evaluation",
)
_WORLD_FIELDS: Final = (
    "image_path",
    "source_id",
    "meters_per_pixel",
    "local_origin",
    "invalid_pixel_rule",
    "valid_region_m",
)
_DRONE_FIELDS: Final = (
    "initial_position_m",
    "altitude_m",
    "speed_mps",
    "battery_capacity_wh",
    "flight_power_w",
)
_TRAJECTORY_FIELDS: Final = ("type", "waypoints_m", "region_m", "lane_spacing_m")
_CAMERA_FIELDS: Final = (
    "projection",
    "footprint_width_m",
    "footprint_height_m",
    "output_width_px",
    "output_height_px",
)
_SIMULATION_FIELDS: Final = (
    "observation_interval_s",
    "fallback_config_id",
    "matching_iou_threshold",
    "network",
    "network_trace",
    "telemetry",
    "policy_execution",
)
_POLICY_EXECUTION_FIELDS: Final = (
    "location",
    "latency_s_per_decision",
    "energy_j_per_decision",
    "state_mb_per_decision",
    "action_mb_per_decision",
    "server_compute_s",
    "energy_j_per_mb",
    "max_loss_frac",
    "on_lost_decision",
)
_POLICY_EXECUTION_LOCATIONS: Final = ("onboard", "server")
_POLICY_EXECUTION_ONBOARD_FIELDS: Final = ("latency_s_per_decision", "energy_j_per_decision")
_POLICY_EXECUTION_SERVER_FIELDS: Final = (
    "state_mb_per_decision",
    "action_mb_per_decision",
    "server_compute_s",
    "energy_j_per_mb",
    "max_loss_frac",
    "on_lost_decision",
)
_ON_LOST_DECISION_MODES: Final = ("hold", "fallback")
_TELEMETRY_FIELDS: Final = (
    "interval_s",
    "report_mb",
    "energy_j_per_mb",
    "max_loss_frac",
    "evidence_mb_per_detection",
    "stream_mb_per_observation",
)
_NETWORK_FIELDS: Final = ("bandwidth_mbps", "rtt_ms", "packet_loss_frac")
_NETWORK_REGIME_FIELDS: Final = (
    "regime_id",
    "start_s",
    "uplink_mbps",
    "downlink_mbps",
    "rtt_ms",
    "packet_loss_frac",
)
_OBJECT_FIELDS: Final = (
    "object_id",
    "class_id",
    "is_target",
    "render_mode",
    "asset_id",
    "position_m",
    "width_m",
    "height_m",
    "rotation_deg",
    "z_order",
    "opacity",
    "brightness_factor",
    "appearance",
)
_RENDER_MODES: Final = ("procedural_marker", "image_asset")
_APPEARANCE_FIELDS: Final = ("shape", "body_rgb", "stripe_rgb", "stripe", "alpha")
_EXECUTOR_FIELDS: Final = (
    "config_id",
    "model_strategy_id",
    "kind",
    "mission_latency_s",
    "energy_j_per_call",
    "communication_mb_per_call",
    "quality_tier",
    "parameters",
)
_CONTRACT_FIELDS: Final = (
    "contract_id",
    "quality_metric",
    "quality_operator",
    "quality_threshold",
    "deadline_s",
    "communication_budget_mb",
    "min_final_battery_frac",
    "privacy_level",
)

#: The executor kinds V2 ships. The two heuristic kinds are the V2.0 test backends; the
#: torch kind is the V2.1 real-model seam and needs the optional [v2-real-models] extra;
#: ``simulated_remote`` (V3 P1) is the deployment-boundary remote path — its backend is
#: one of the other kinds run "server-side"; ``simulated_split`` runs the model's head
#: onboard and ships intermediate features to the simulated server for the tail;
#: ``pretrained_split`` runs a model published already split by prior research
#: (mandatory literature provenance — see ``aerointentbench.v2.presplit``).
_EXECUTOR_KINDS: Final = (
    "fast_weak",
    "slow_strong",
    "torch_semantic_segmentation",
    "torch_instance_segmentation",
    "onnx_semantic_segmentation",
    "simulated_remote",
    "simulated_split",
    "pretrained_split",
)

#: The kinds that offload work across the simulated link. Everything else is onboard;
#: offload kinds may not serve as another offload config's backend.
_OFFLOAD_KINDS: Final = ("simulated_remote", "simulated_split", "pretrained_split")

#: Required entries in ``parameters`` for a simulated_remote executor. Everything else
#: (encoding/queue/energy coefficients, fallback_config_id, download size, loss ceiling)
#: has documented defaults in ``aerointentbench.v2.remote.REMOTE_PARAMETER_DEFAULTS``.
_REMOTE_REQUIRED_PARAMETERS: Final = ("backend_kind", "remote_compute_s", "timeout_s")

#: Required entries in ``parameters`` for a simulated_split executor: the remote set
#: plus the cut identity and the head's onboard latency (its energy is the config's
#: ``energy_j_per_call``). ``feature_dtype`` defaults in
#: ``aerointentbench.v2.split.SPLIT_PARAMETER_DEFAULTS``.
_SPLIT_REQUIRED_PARAMETERS: Final = (
    "backend_kind",
    "split_cut",
    "head_latency_s",
    "remote_compute_s",
    "timeout_s",
)

#: Required entries in ``parameters`` for a pretrained_split executor: the shared
#: offload timings plus the backend selector and the MANDATORY split provenance —
#: a predefined split must be citable, never discovered by this benchmark
#: (``aerointentbench.v2.presplit.SplitSpec``).
_PRESPLIT_REQUIRED_PARAMETERS: Final = (
    "split_backend",
    "split_source",
    "split_source_reference",
    "split_location",
    "head_latency_s",
    "remote_compute_s",
    "timeout_s",
)
_TRAJECTORY_TYPES: Final = ("polyline", "lawnmower")

#: Required entries in ``parameters`` for a torch_semantic_segmentation executor. Kept in
#: the scalar ``parameters`` map -- its documented extension-point role -- rather than as
#: new top-level fields, so V2.0 configs are not reinterpreted.
_TORCH_REQUIRED_PARAMETERS: Final = (
    "model_id",
    "weights_id",
    "task_class",
    "input_width_px",
    "input_height_px",
    "probability_threshold",
    "device",
    "dtype",
    "latency_mode",
)
_TORCH_LATENCY_MODES: Final = ("measured", "configured")
_TORCH_DTYPES: Final = ("float32", "float16")


@dataclass(frozen=True, slots=True)
class WorldSpec:
    image_path: str
    source_id: str
    #: Ground meters per source-raster pixel. For a GeoTIFF the loader may verify this
    #: against the file's transform; for a plain image it is the only scale there is.
    meters_per_pixel: float
    invalid_pixel_rule: str = "all_zero"
    #: Optional conservative rectangle (x0, y0, x1, y1) in meters that the mission must
    #: stay inside. Trajectory, objects, and footprints are validated against it.
    valid_region_m: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class DroneSpec:
    initial_position_m: tuple[float, float]
    altitude_m: float
    speed_mps: float
    battery_capacity_wh: float
    flight_power_w: float


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    type: str
    waypoints_m: tuple[tuple[float, float], ...] = ()
    region_m: tuple[float, float, float, float] | None = None
    lane_spacing_m: float | None = None


@dataclass(frozen=True, slots=True)
class CameraSpec:
    projection: str
    footprint_width_m: float
    footprint_height_m: float
    output_width_px: int
    output_height_px: int


@dataclass(frozen=True, slots=True)
class TelemetrySpec:
    """Periodic status uplink the drone sends regardless of where inference runs.

    Models the mission-progress reports (position, battery, status, detection summary)
    a real search UAV streams to its ground station even when perception is fully
    local. Reports are asynchronous and never block the perception loop (no latency
    charge); each attempt at its scheduled mission time uses the capture-time network
    sample — sent when the uplink is up and loss is within ``max_loss_frac``, silently
    lost otherwise (lost reports transfer nothing and cost nothing). Sent report MB
    count against the mission communication budget; report energy is charged to the
    battery, ledgered separately from remote-inference transfer energy.
    """

    interval_s: float
    report_mb: float
    energy_j_per_mb: float
    max_loss_frac: float
    #: Evidence traffic per predicted detection (0.0 = periodic status reports only).
    #: When an observation completes with N predicted components, one evidence
    #: transmission of ``N x evidence_mb_per_detection`` MB is attempted at completion
    #: time — the user sees what the drone believes it found, so false positives spend
    #: real communication too. Derived from the drone's own predictions, never ground
    #: truth.
    evidence_mb_per_detection: float = 0.0
    #: Continuous observation downlink (0.0 = off): one frame of this size is attempted
    #: at every capture tick, so the control center watches what the drone sees. This is
    #: imagery leaving the vehicle — however downscaled, compression is not
    #: de-identification — so a scenario may only enable it under ``remote_allowed``
    #: (validated at load). Frames lost during outages are the operator's blind time.
    stream_mb_per_observation: float = 0.0


@dataclass(frozen=True, slots=True)
class PolicyExecutionSpec:
    """Where the selection policy runs, and what each decision costs.

    The benchmark historically treated the policy's own execution as free. This block
    makes the decision-maker a resource consumer like everything else, without touching
    the frozen ``Policy`` interface — the runner charges the declared costs around each
    ``select_config`` call. The location is deployment configuration, never a policy
    action (the decider cannot choose where the decider runs).

    ``onboard``: each decision advances the mission clock by ``latency_s_per_decision``
    (a slow policy skips capture slots exactly like a slow executor) and charges
    ``energy_j_per_decision`` to the battery. Rule-family policies are microseconds and
    honestly declare 0.0; a future LLM policy declares its measured cost.

    ``server``: each decision is a round trip at the capture-time network sample —
    state up, action down. Sent MB share the contract's communication budget; transfer
    energy is charged like telemetry. Decision latency is DERIVED, never configured
    opaquely: state_mb/uplink + rtt + server_compute_s + action_mb/downlink. When the
    link is down (either direction) or loss exceeds ``max_loss_frac``, the decision is
    LOST — lost-and-free like telemetry — and the drone acts on ``on_lost_decision``:
    ``hold`` keeps the current configuration (first slot: the scenario fallback),
    ``fallback`` drops to the scenario fallback. Lost decisions are counted: a
    server-hosted policy is blind exactly when the mission is hardest.
    """

    location: str
    #: Onboard costs (configured mission-scale values; label their provenance).
    latency_s_per_decision: float = 0.0
    energy_j_per_decision: float = 0.0
    #: Server round-trip terms.
    state_mb_per_decision: float = 0.0
    action_mb_per_decision: float = 0.0
    server_compute_s: float = 0.0
    energy_j_per_mb: float = 0.0
    max_loss_frac: float = 0.0
    on_lost_decision: str = "hold"


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    observation_interval_s: float
    #: Safe fallback when the policy's action is invalid. Declared explicitly -- a V2
    #: scenario never inherits the V1 legacy CFG_LOCAL_LIGHT assumption.
    fallback_config_id: str
    #: IoU at which a predicted component counts as matching a ground-truth instance.
    matching_iou_threshold: float
    #: Constant policy-visible network conditions, used when no trace is declared.
    network: tuple[float, float, float]  # bandwidth_mbps, rtt_ms, packet_loss_frac
    #: Optional time-varying named network regimes (V3 P1). When present, the runner
    #: samples this piecewise-constant trace instead of the constant ``network``.
    network_trace: tuple[NetworkRegime, ...] | None = None
    #: Optional periodic ground-station telemetry. Absent means no telemetry traffic —
    #: existing scenarios and their results are untouched.
    telemetry: TelemetrySpec | None = None
    #: Optional policy-execution cost model. Absent means the historical free-policy
    #: behaviour — existing scenarios and their results are untouched.
    policy_execution: PolicyExecutionSpec | None = None


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    object_id: str
    class_id: str
    is_target: bool
    position_m: tuple[float, float]
    width_m: float
    height_m: float
    rotation_deg: float
    z_order: int
    appearance: Mapping[str, Any] = field(default_factory=dict)
    #: How this object is drawn: the V2.0 procedural marker (default) or a V2.2 image
    #: asset resolved through the scenario's assets_manifest by ``asset_id``.
    render_mode: str = "procedural_marker"
    asset_id: str | None = None
    #: Compositing opacity of the whole object (multiplies the asset alpha). RGB only;
    #: GT geometry is unaffected.
    opacity: float = 1.0
    #: Deterministic RGB-only appearance transform (1.0 = unchanged). Never touches GT.
    brightness_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class ExecutorConfigSpec:
    config_id: str
    model_strategy_id: str
    kind: str
    #: The latency the *mission* experiences. Simulated/configured, never measured.
    mission_latency_s: float
    energy_j_per_call: float
    communication_mb_per_call: float
    quality_tier: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class V2Scenario:
    scenario_id: str
    random_seed: int
    world: WorldSpec
    drone: DroneSpec
    trajectory: TrajectorySpec
    camera: CameraSpec
    simulation: SimulationSpec
    objects: tuple[ObjectSpec, ...]
    executor_configs: tuple[ExecutorConfigSpec, ...]
    contract: Contract
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: Path to the target-asset manifest (required when any object uses image_asset).
    assets_manifest: str | None = None
    #: What results from this scenario may claim (see _EVALUATION_PURPOSES).
    evaluation_purpose: str | None = None

    @property
    def config_ids(self) -> tuple[str, ...]:
        return tuple(spec.config_id for spec in self.executor_configs)

    @property
    def targets(self) -> tuple[ObjectSpec, ...]:
        return tuple(obj for obj in self.objects if obj.is_target)

    @property
    def distractors(self) -> tuple[ObjectSpec, ...]:
        return tuple(obj for obj in self.objects if not obj.is_target)


def load_scenario(path: Path) -> V2Scenario:
    """Load and strictly validate a V2 scenario file."""
    payload = read_json_object(path)
    context = f"{path.name} -> V2Scenario"

    declared = payload.get("scenario_schema_version")
    if declared != SCENARIO_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{context}: scenario_schema_version {declared!r} is not supported; this "
            f"release reads {SCENARIO_SCHEMA_VERSION!r} only"
        )

    reader = DocumentReader(payload, context=context, allowed_fields=_TOP_FIELDS)
    world = _read_world(reader.get_object("world", allowed_fields=_WORLD_FIELDS))
    drone = _read_drone(reader.get_object("drone", allowed_fields=_DRONE_FIELDS))
    trajectory = _read_trajectory(
        reader.get_object("trajectory", allowed_fields=_TRAJECTORY_FIELDS)
    )
    camera = _read_camera(reader.get_object("camera", allowed_fields=_CAMERA_FIELDS))
    simulation = _read_simulation(
        reader.get_object("simulation", allowed_fields=_SIMULATION_FIELDS)
    )
    objects = _read_objects(reader)
    executors = _read_executors(reader)
    contract = _read_contract(
        reader.get_object("mission_contract", allowed_fields=_CONTRACT_FIELDS)
    )

    evaluation_purpose = reader.get_optional_str("evaluation_purpose")
    if evaluation_purpose is not None and evaluation_purpose not in _EVALUATION_PURPOSES:
        raise SchemaValidationError(
            f"{context}: evaluation_purpose {evaluation_purpose!r} must be one of "
            f"{list(_EVALUATION_PURPOSES)}"
        )

    scenario = V2Scenario(
        scenario_id=reader.get_str("scenario_id"),
        random_seed=reader.get_int("random_seed"),
        world=world,
        drone=drone,
        trajectory=trajectory,
        camera=camera,
        simulation=simulation,
        objects=objects,
        executor_configs=executors,
        contract=contract,
        provenance=dict(payload.get("provenance") or {}),
        assets_manifest=reader.get_optional_str("assets_manifest"),
        evaluation_purpose=evaluation_purpose,
    )
    _cross_validate(scenario, context)
    return scenario


# --- section readers -------------------------------------------------------------------------


def _read_world(reader: DocumentReader) -> WorldSpec:
    rule = reader.get_optional_str("invalid_pixel_rule") or "all_zero"
    if rule not in ("all_zero", "dataset_mask"):
        raise SchemaValidationError(
            f"{reader.context}: invalid_pixel_rule {rule!r} must be 'all_zero' or 'dataset_mask'"
        )
    return WorldSpec(
        image_path=reader.get_str("image_path"),
        source_id=reader.get_str("source_id"),
        meters_per_pixel=reader.get_float("meters_per_pixel", exclusive_minimum=0.0),
        invalid_pixel_rule=rule,
        valid_region_m=_optional_rect(reader, "valid_region_m"),
    )


def _read_drone(reader: DocumentReader) -> DroneSpec:
    return DroneSpec(
        initial_position_m=_point(reader, "initial_position_m"),
        altitude_m=reader.get_float("altitude_m", exclusive_minimum=0.0),
        speed_mps=reader.get_float("speed_mps", exclusive_minimum=0.0),
        battery_capacity_wh=reader.get_float("battery_capacity_wh", exclusive_minimum=0.0),
        flight_power_w=reader.get_float("flight_power_w", minimum=0.0),
    )


def _read_trajectory(reader: DocumentReader) -> TrajectorySpec:
    kind = reader.get_str("type")
    if kind not in _TRAJECTORY_TYPES:
        raise SchemaValidationError(
            f"{reader.context}: trajectory type {kind!r} must be one of {list(_TRAJECTORY_TYPES)}"
        )
    if kind == "polyline":
        raw = reader.get_passthrough("waypoints_m")
        if not isinstance(raw, list) or len(raw) < 2:
            raise SchemaValidationError(
                f"{reader.context}: a polyline trajectory needs 'waypoints_m' with >= 2 points"
            )
        waypoints = tuple(
            _as_point(p, reader.context, f"waypoints_m[{i}]") for i, p in enumerate(raw)
        )
        return TrajectorySpec(type=kind, waypoints_m=waypoints)
    region = _optional_rect(reader, "region_m")
    spacing = reader.get_passthrough("lane_spacing_m")
    if region is None or spacing is None:
        raise SchemaValidationError(
            f"{reader.context}: a lawnmower trajectory needs 'region_m' and 'lane_spacing_m'"
        )
    spacing_f = reader.get_float("lane_spacing_m", exclusive_minimum=0.0)
    return TrajectorySpec(type=kind, region_m=region, lane_spacing_m=spacing_f)


def _read_camera(reader: DocumentReader) -> CameraSpec:
    projection = reader.get_str("projection")
    if projection != "orthographic":
        raise SchemaValidationError(
            f"{reader.context}: projection {projection!r} is not supported; V2 supports "
            "'orthographic' only"
        )
    return CameraSpec(
        projection=projection,
        footprint_width_m=reader.get_float("footprint_width_m", exclusive_minimum=0.0),
        footprint_height_m=reader.get_float("footprint_height_m", exclusive_minimum=0.0),
        output_width_px=reader.get_int("output_width_px", minimum=8),
        output_height_px=reader.get_int("output_height_px", minimum=8),
    )


def _read_simulation(reader: DocumentReader) -> SimulationSpec:
    network = reader.get_object("network", allowed_fields=_NETWORK_FIELDS)
    trace: tuple[NetworkRegime, ...] | None = None
    if reader.get_passthrough("network_trace") is not None:
        regimes = [
            NetworkRegime(
                regime_id=entry.get_str("regime_id"),
                start_s=entry.get_float("start_s", minimum=0.0),
                uplink_mbps=entry.get_float("uplink_mbps", minimum=0.0),
                downlink_mbps=entry.get_float("downlink_mbps", minimum=0.0),
                rtt_ms=entry.get_float("rtt_ms", minimum=0.0),
                packet_loss_frac=entry.get_fraction("packet_loss_frac"),
            )
            for entry in reader.get_object_list(
                "network_trace", allowed_fields=_NETWORK_REGIME_FIELDS
            )
        ]
        trace = tuple(regimes)
    telemetry: TelemetrySpec | None = None
    if reader.get_passthrough("telemetry") is not None:
        telemetry_reader = reader.get_object("telemetry", allowed_fields=_TELEMETRY_FIELDS)
        evidence_mb = 0.0
        if telemetry_reader.get_passthrough("evidence_mb_per_detection") is not None:
            evidence_mb = telemetry_reader.get_float("evidence_mb_per_detection", minimum=0.0)
        stream_mb = 0.0
        if telemetry_reader.get_passthrough("stream_mb_per_observation") is not None:
            stream_mb = telemetry_reader.get_float("stream_mb_per_observation", minimum=0.0)
        telemetry = TelemetrySpec(
            interval_s=telemetry_reader.get_float("interval_s", exclusive_minimum=0.0),
            report_mb=telemetry_reader.get_float("report_mb", minimum=0.0),
            energy_j_per_mb=telemetry_reader.get_float("energy_j_per_mb", minimum=0.0),
            max_loss_frac=telemetry_reader.get_fraction("max_loss_frac"),
            evidence_mb_per_detection=evidence_mb,
            stream_mb_per_observation=stream_mb,
        )
    policy_execution: PolicyExecutionSpec | None = None
    if reader.get_passthrough("policy_execution") is not None:
        policy_execution = _read_policy_execution(reader)
    return SimulationSpec(
        observation_interval_s=reader.get_float("observation_interval_s", exclusive_minimum=0.0),
        fallback_config_id=reader.get_str("fallback_config_id"),
        matching_iou_threshold=reader.get_float(
            "matching_iou_threshold", exclusive_minimum=0.0, maximum=1.0
        ),
        network=(
            network.get_float("bandwidth_mbps", minimum=0.0),
            network.get_float("rtt_ms", minimum=0.0),
            network.get_fraction("packet_loss_frac"),
        ),
        network_trace=trace,
        telemetry=telemetry,
        policy_execution=policy_execution,
    )


def _read_policy_execution(reader: DocumentReader) -> PolicyExecutionSpec:
    block = reader.get_object("policy_execution", allowed_fields=_POLICY_EXECUTION_FIELDS)
    location = block.get_str("location")
    if location not in _POLICY_EXECUTION_LOCATIONS:
        raise SchemaValidationError(
            f"{block.context}: location {location!r} is not one of "
            f"{list(_POLICY_EXECUTION_LOCATIONS)}"
        )
    # A location's spec may only declare its own cost terms — a stray field from the
    # other location is a spec mistake, not a default to ignore silently.
    foreign = (
        _POLICY_EXECUTION_SERVER_FIELDS
        if location == "onboard"
        else _POLICY_EXECUTION_ONBOARD_FIELDS
    )
    declared = [f for f in foreign if block.get_passthrough(f) is not None]
    if declared:
        raise SchemaValidationError(
            f"{block.context}: location {location!r} does not use field(s) {declared}"
        )
    if location == "onboard":
        return PolicyExecutionSpec(
            location=location,
            latency_s_per_decision=(
                block.get_float("latency_s_per_decision", minimum=0.0)
                if block.get_passthrough("latency_s_per_decision") is not None
                else 0.0
            ),
            energy_j_per_decision=(
                block.get_float("energy_j_per_decision", minimum=0.0)
                if block.get_passthrough("energy_j_per_decision") is not None
                else 0.0
            ),
        )
    on_lost = (
        block.get_str("on_lost_decision")
        if block.get_passthrough("on_lost_decision") is not None
        else "hold"
    )
    if on_lost not in _ON_LOST_DECISION_MODES:
        raise SchemaValidationError(
            f"{block.context}: on_lost_decision {on_lost!r} is not one of "
            f"{list(_ON_LOST_DECISION_MODES)}"
        )
    return PolicyExecutionSpec(
        location=location,
        state_mb_per_decision=block.get_float("state_mb_per_decision", exclusive_minimum=0.0),
        action_mb_per_decision=block.get_float("action_mb_per_decision", exclusive_minimum=0.0),
        server_compute_s=block.get_float("server_compute_s", minimum=0.0),
        energy_j_per_mb=block.get_float("energy_j_per_mb", minimum=0.0),
        max_loss_frac=block.get_fraction("max_loss_frac"),
        on_lost_decision=on_lost,
    )


def _read_objects(reader: DocumentReader) -> tuple[ObjectSpec, ...]:
    objects: list[ObjectSpec] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("objects", allowed_fields=_OBJECT_FIELDS):
        object_id = entry.get_str("object_id")
        if object_id in seen:
            raise SchemaValidationError(f"{entry.context}: duplicate object_id {object_id!r}")
        seen.add(object_id)

        render_mode = entry.get_optional_str("render_mode") or "procedural_marker"
        if render_mode not in _RENDER_MODES:
            raise SchemaValidationError(
                f"{entry.context}: render_mode {render_mode!r} must be one of {list(_RENDER_MODES)}"
            )
        asset_id = entry.get_optional_str("asset_id")
        if render_mode == "image_asset":
            if asset_id is None:
                raise SchemaValidationError(
                    f"{entry.context}: render_mode 'image_asset' requires 'asset_id'"
                )
            appearance: dict[str, Any] = {}
            if entry.get_passthrough("appearance") is not None:
                raise SchemaValidationError(
                    f"{entry.context}: 'appearance' is the procedural-marker block; an "
                    "image_asset object styles itself via opacity/brightness_factor"
                )
        else:
            if asset_id is not None:
                raise SchemaValidationError(
                    f"{entry.context}: asset_id is only valid with render_mode 'image_asset'"
                )
            appearance = _read_appearance(
                entry.get_object("appearance", allowed_fields=_APPEARANCE_FIELDS)
            )

        objects.append(
            ObjectSpec(
                object_id=object_id,
                class_id=entry.get_str("class_id"),
                is_target=entry.get_bool("is_target"),
                position_m=_point(entry, "position_m"),
                width_m=entry.get_float("width_m", exclusive_minimum=0.0),
                height_m=entry.get_float("height_m", exclusive_minimum=0.0),
                rotation_deg=entry.get_float("rotation_deg", minimum=-360.0, maximum=360.0),
                z_order=entry.get_int("z_order"),
                appearance=appearance,
                render_mode=render_mode,
                asset_id=asset_id,
                opacity=entry.get_optional_float("opacity", default=1.0, minimum=0.1, maximum=1.0),
                brightness_factor=entry.get_optional_float(
                    "brightness_factor", default=1.0, minimum=0.2, maximum=3.0
                ),
            )
        )
    return tuple(objects)


def _read_appearance(reader: DocumentReader) -> dict[str, Any]:
    shape = reader.get_optional_str("shape") or "rect"
    if shape not in ("rect", "ellipse"):
        raise SchemaValidationError(
            f"{reader.context}: shape {shape!r} must be 'rect' or 'ellipse'"
        )
    return {
        "shape": shape,
        "body_rgb": _rgb(reader, "body_rgb"),
        "stripe_rgb": _rgb(reader, "stripe_rgb") if reader.get_passthrough("stripe_rgb") else None,
        "stripe": (
            bool(reader.get_bool("stripe"))
            if reader.get_passthrough("stripe") is not None
            else False
        ),
        "alpha": reader.get_optional_float("alpha", default=1.0, minimum=0.1, maximum=1.0),
    }


def _read_executors(reader: DocumentReader) -> tuple[ExecutorConfigSpec, ...]:
    executors: list[ExecutorConfigSpec] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("executor_configs", allowed_fields=_EXECUTOR_FIELDS):
        config_id = entry.get_str("config_id")
        if config_id in seen:
            raise SchemaValidationError(f"{entry.context}: duplicate config_id {config_id!r}")
        seen.add(config_id)
        kind = entry.get_str("kind")
        if kind not in _EXECUTOR_KINDS:
            raise SchemaValidationError(
                f"{entry.context}: executor kind {kind!r} must be one of {list(_EXECUTOR_KINDS)}"
            )
        parameters = entry.get_scalar_mapping("parameters")
        if kind == "torch_semantic_segmentation":
            _validate_torch_parameters(parameters, entry.context)
        if kind == "torch_instance_segmentation":
            from aerointentbench.v2.instance_models import validate_instance_parameters

            validate_instance_parameters(parameters, entry.context)
        if kind == "onnx_semantic_segmentation":
            from aerointentbench.v2.onnx_models import validate_onnx_parameters

            validate_onnx_parameters(parameters, entry.context)
        if kind == "simulated_remote":
            _validate_remote_parameters(parameters, entry.context)
        if kind == "simulated_split":
            _validate_split_parameters(parameters, entry.context)
        if kind == "pretrained_split":
            _validate_presplit_parameters(parameters, entry.context)
        executors.append(
            ExecutorConfigSpec(
                config_id=config_id,
                model_strategy_id=entry.get_str("model_strategy_id"),
                kind=kind,
                mission_latency_s=entry.get_float("mission_latency_s", exclusive_minimum=0.0),
                energy_j_per_call=entry.get_float("energy_j_per_call", minimum=0.0),
                communication_mb_per_call=entry.get_float("communication_mb_per_call", minimum=0.0),
                quality_tier=entry.get_str("quality_tier"),
                parameters=parameters,
            )
        )
    if len(executors) < 2:
        raise SchemaValidationError(
            f"{reader.context}: at least two executor_configs are required, got {len(executors)}"
        )
    return tuple(executors)


def _read_contract(reader: DocumentReader) -> Contract:
    """Build a V1 ``Contract`` from the scenario's nested mission_contract.

    Reusing the frozen V1 dataclass keeps the policy interface identical -- a V1 policy
    receives exactly the contract type it already knows -- without touching the V1 file
    schema. Task identity is fixed for V2's single visual task; ``privacy_level`` is
    declarable since V3 P1 (default ``remote_allowed``, the historical V2 behaviour).
    """
    declared_privacy = reader.get_optional_str("privacy_level")
    try:
        privacy = (
            PrivacyLevel(declared_privacy)
            if declared_privacy is not None
            else PrivacyLevel.REMOTE_ALLOWED
        )
    except ValueError:
        raise SchemaValidationError(
            f"{reader.context}: privacy_level {declared_privacy!r} must be one of "
            f"{[level.value for level in PrivacyLevel]}"
        ) from None
    return Contract(
        contract_id=reader.get_str("contract_id"),
        task_id="HUMAN_SEARCH_SEGMENTATION",
        evidence_type="instance_mask_set",
        quality_metric=reader.get_str("quality_metric"),
        quality_operator=ComparisonOperator(reader.get_str("quality_operator")),
        quality_threshold=reader.get_fraction("quality_threshold"),
        deadline_s=reader.get_float("deadline_s", exclusive_minimum=0.0),
        communication_budget_mb=reader.get_float("communication_budget_mb", minimum=0.0),
        min_final_battery_frac=reader.get_fraction("min_final_battery_frac"),
        privacy_level=privacy,
    )


def _validate_torch_parameters(parameters: Mapping[str, Any], context: str) -> None:
    """Validate the model-strategy parameters of a real-model executor at load time.

    Failing here -- with the exact missing or invalid key -- beats failing after the
    world has been opened and a mission is halfway constructed.
    """
    missing = [key for key in _TORCH_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: a torch_semantic_segmentation executor requires parameters "
            f"{list(_TORCH_REQUIRED_PARAMETERS)}; missing {missing}"
        )
    if parameters["latency_mode"] not in _TORCH_LATENCY_MODES:
        raise SchemaValidationError(
            f"{context}: latency_mode {parameters['latency_mode']!r} must be one of "
            f"{list(_TORCH_LATENCY_MODES)}"
        )
    if parameters["dtype"] not in _TORCH_DTYPES:
        raise SchemaValidationError(
            f"{context}: dtype {parameters['dtype']!r} must be one of {list(_TORCH_DTYPES)}"
        )
    for key in ("input_width_px", "input_height_px"):
        value = parameters[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 32:
            raise SchemaValidationError(f"{context}: {key} must be an integer >= 32, got {value!r}")
    threshold = parameters["probability_threshold"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not (0.0 < float(threshold) < 1.0)
    ):
        raise SchemaValidationError(
            f"{context}: probability_threshold must be in (0, 1), got {threshold!r}"
        )


def _validate_remote_parameters(parameters: Mapping[str, Any], context: str) -> None:
    """Validate a simulated_remote executor's model-strategy parameters at load time."""
    missing = [key for key in _REMOTE_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: a simulated_remote executor requires parameters "
            f"{list(_REMOTE_REQUIRED_PARAMETERS)}; missing {missing}"
        )
    backend = parameters["backend_kind"]
    local_kinds = tuple(k for k in _EXECUTOR_KINDS if k not in _OFFLOAD_KINDS)
    if backend not in local_kinds:
        raise SchemaValidationError(
            f"{context}: backend_kind {backend!r} must be one of {list(local_kinds)} "
            "(a remote backend is an existing model kind run server-side)"
        )
    for key in ("remote_compute_s", "timeout_s"):
        value = parameters[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise SchemaValidationError(
                f"{context}: {key} must be a positive number, got {value!r}"
            )


def _validate_split_parameters(parameters: Mapping[str, Any], context: str) -> None:
    """Validate a simulated_split executor's model-strategy parameters at load time."""
    missing = [key for key in _SPLIT_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: a simulated_split executor requires parameters "
            f"{list(_SPLIT_REQUIRED_PARAMETERS)}; missing {missing}"
        )
    backend = parameters["backend_kind"]
    local_kinds = tuple(k for k in _EXECUTOR_KINDS if k not in _OFFLOAD_KINDS)
    if backend not in local_kinds:
        raise SchemaValidationError(
            f"{context}: split backend_kind {backend!r} must be one of {list(local_kinds)} "
            "(the split head and tail partition an existing model kind)"
        )
    _validate_positive_timings(parameters, context)
    dtype = parameters.get("feature_dtype", "float16")
    if dtype not in ("float32", "float16", "uint8"):
        raise SchemaValidationError(
            f"{context}: feature_dtype {dtype!r} must be one of ['float32', 'float16', 'uint8']"
        )


def _validate_positive_timings(parameters: Mapping[str, Any], context: str) -> None:
    for key in ("head_latency_s", "remote_compute_s", "timeout_s"):
        value = parameters[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise SchemaValidationError(
                f"{context}: {key} must be a positive number, got {value!r}"
            )


def _validate_presplit_parameters(parameters: Mapping[str, Any], context: str) -> None:
    """Validate a pretrained_split executor's model-strategy parameters at load time.

    Provenance is part of the schema: a predefined split without a citable source
    is rejected here, before anything runs. Backend-specific requirements (e.g.
    the sc2 checkpoint path) are validated by the backend at build time, mirroring
    the torch kind.
    """
    missing = [key for key in _PRESPLIT_REQUIRED_PARAMETERS if key not in parameters]
    if missing:
        raise SchemaValidationError(
            f"{context}: a pretrained_split executor requires parameters "
            f"{list(_PRESPLIT_REQUIRED_PARAMETERS)}; missing {missing}"
        )
    from aerointentbench.v2.presplit import PRESPLIT_BACKENDS, SPLIT_SOURCES

    backend = parameters["split_backend"]
    if backend not in PRESPLIT_BACKENDS:
        raise SchemaValidationError(
            f"{context}: split_backend {backend!r} must be one of {list(PRESPLIT_BACKENDS)}"
        )
    source = parameters["split_source"]
    if source not in SPLIT_SOURCES:
        raise SchemaValidationError(
            f"{context}: split_source {source!r} must be one of {list(SPLIT_SOURCES)} — a "
            "predefined split is adopted from prior work, never discovered by this benchmark"
        )
    for key in ("split_source_reference", "split_location"):
        value = parameters[key]
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(
                f"{context}: {key} must be a non-empty string documenting the split's origin"
            )
    _validate_positive_timings(parameters, context)


# --- cross-document checks -------------------------------------------------------------------


def _cross_validate(scenario: V2Scenario, context: str) -> None:
    if scenario.simulation.fallback_config_id not in scenario.config_ids:
        raise SchemaValidationError(
            f"{context}: fallback_config_id {scenario.simulation.fallback_config_id!r} is not "
            f"one of the executor_configs {list(scenario.config_ids)}"
        )
    if scenario.contract.quality_metric not in ("target_recall", "detection_precision"):
        raise SchemaValidationError(
            f"{context}: quality_metric {scenario.contract.quality_metric!r} is not an "
            "empirical V2 metric; use 'target_recall' (canonical) or 'detection_precision'"
        )
    telemetry = scenario.simulation.telemetry
    if (
        telemetry is not None
        and telemetry.stream_mb_per_observation > 0.0
        and scenario.contract.privacy_level is not PrivacyLevel.REMOTE_ALLOWED
    ):
        raise SchemaValidationError(
            f"{context}: stream_mb_per_observation streams imagery off the vehicle, which "
            f"privacy_level {scenario.contract.privacy_level.value!r} forbids (compression is "
            "not de-identification); enable the stream only under 'remote_allowed'"
        )
    policy_execution = scenario.simulation.policy_execution
    if (
        policy_execution is not None
        and policy_execution.location == "server"
        and scenario.contract.privacy_level is PrivacyLevel.LOCAL_ONLY
    ):
        raise SchemaValidationError(
            f"{context}: policy_execution location 'server' sends mission state off the "
            "vehicle for offboard decision-making, which privacy_level 'local_only' "
            "forbids; use 'features_only' or 'remote_allowed'"
        )
    region = scenario.world.valid_region_m
    if region is not None:
        x0, y0, x1, y1 = region
        if not (x1 > x0 and y1 > y0):
            raise SchemaValidationError(f"{context}: valid_region_m {region} is not a proper rect")
        for obj in scenario.objects:
            x, y = obj.position_m
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                raise SchemaValidationError(
                    f"{context}: object {obj.object_id!r} at {obj.position_m} lies outside "
                    f"valid_region_m {region}"
                )

    image_objects = [obj for obj in scenario.objects if obj.render_mode == "image_asset"]
    if image_objects and scenario.assets_manifest is None:
        raise SchemaValidationError(
            f"{context}: object(s) {[o.object_id for o in image_objects]} use render_mode "
            "'image_asset' but the scenario declares no 'assets_manifest'"
        )

    if scenario.simulation.network_trace is not None:
        from aerointentbench.v2.network import V2NetworkModel

        V2NetworkModel(scenario.simulation.network_trace)  # fails loudly at load time

    local_ids = {s.config_id for s in scenario.executor_configs if s.kind != "simulated_remote"}
    for spec in scenario.executor_configs:
        if spec.kind != "simulated_remote":
            continue
        fallback = spec.parameters.get("fallback_config_id")
        if fallback is not None and fallback not in local_ids:
            raise SchemaValidationError(
                f"{context}: remote config {spec.config_id!r} names fallback_config_id "
                f"{fallback!r}, which is not a local executor_config in this scenario"
            )
    if scenario.simulation.fallback_config_id not in local_ids:
        raise SchemaValidationError(
            f"{context}: the scenario-level fallback_config_id "
            f"{scenario.simulation.fallback_config_id!r} must be a local configuration -- "
            "the safe fallback may never depend on the network"
        )


# --- low-level field helpers -----------------------------------------------------------------


def _as_point(raw: object, context: str, name: str) -> tuple[float, float]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
    ):
        raise SchemaValidationError(f"{context}: {name} must be [x_m, y_m], got {raw!r}")
    return (float(raw[0]), float(raw[1]))


def _point(reader: DocumentReader, key: str) -> tuple[float, float]:
    return _as_point(reader.get_passthrough(key), reader.context, key)


def _optional_rect(reader: DocumentReader, key: str) -> tuple[float, float, float, float] | None:
    raw = reader.get_passthrough(key)
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 4
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
    ):
        raise SchemaValidationError(
            f"{reader.context}: {key} must be [x0_m, y0_m, x1_m, y1_m], got {raw!r}"
        )
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _rgb(reader: DocumentReader, key: str) -> tuple[int, int, int] | None:
    raw = reader.get_passthrough(key)
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in raw)
    ):
        raise SchemaValidationError(f"{reader.context}: {key} must be [r, g, b] in 0..255")
    return (int(raw[0]), int(raw[1]), int(raw[2]))
