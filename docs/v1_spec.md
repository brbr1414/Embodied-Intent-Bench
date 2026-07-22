# AeroIntentBench V1 specification

Status: **scaffold committed, implementation in progress.** This document is the
normative reference for V1. Where implementation and this document disagree, this
document is the bug report.

> Every numeric value shown below is a **synthetic example**, not a measurement.

## 1. What the benchmark evaluates

> Can a policy select appropriate perception inference configurations under changing
> battery and network conditions while satisfying a mission contract?

The subject under evaluation is a **configuration-selection policy**. At each decision
step the policy observes a mission contract and a policy-visible runtime state, and
returns one configuration ID from a fixed pool.

**Not evaluated:** path planning, navigation, flight control, open-ended language
understanding, or perception model quality in isolation.

## 2. V1 scope

In scope:

- One vehicle class: UAV, following a **predefined path** at fixed nominal altitude and velocity.
- One perception task family: **segmentation**.
- One mission type: **human search** (`HUMAN_SEARCH_SEGMENTATION`).
- Decision interval: **one second**.
- Policy action: one predefined configuration ID.
- Dynamic runtime variables: battery, network bandwidth, RTT, packet loss, remaining
  deadline, accumulated communication, path progress, accumulated evidence.
- Profile-driven deterministic simulation; local **and** remote configurations.

Explicitly out of scope for V1 — do not add these:

- Real segmentation model execution (interface stub only).
- Physical flight dynamics, navigation policy, Gazebo, PX4, ROS 2, real UAV control.
- Split inference, training pipelines, RL implementations.
- GPU or heavy ML dependencies of any kind.

Model names and hardware profiles remain configurable placeholders.

## 3. The loop

```
JSON specifications
  -> EpisodeRunner
  -> build RuntimeState
  -> Policy.select_config(contract, state, configs) -> config_id
  -> validate action
  -> Executor executes or replays the selected config
  -> update evidence
  -> update time, path progress, energy, battery, communication
  -> check termination
  -> repeat
  -> Evaluator computes episode metrics
  -> aggregate across episodes
```

Per decision step, in order:

1. Build the current `RuntimeState`.
2. Ask the policy for a config ID.
3. Validate the config.
4. Execute or replay the config on the current frame.
5. Update evidence.
6. Update time, path progress, energy, battery, communication.
7. Check termination.

## 4. Schemas

Every specification and result file carries `"schema_version": "1.0"`. V1 supports that
value only; any other value fails loading with a clear validation error. Validation is
strict — unknown fields are rejected rather than ignored, so a misspelling can never
read as an intentional default. See §11 for where migrations would go.

### 4.1 Contract — *what must be achieved*

| Field | Type |
|---|---|
| `contract_id` | string |
| `task_id` | string |
| `evidence_type` | string |
| `quality_metric` | string |
| `quality_operator` | one of `>=`, `<=`, `>`, `<` |
| `quality_threshold` | float |
| `deadline_s` | positive float |
| `communication_budget_mb` | nonnegative float |
| `min_final_battery_frac` | float in [0, 1] |
| `privacy_level` | `local_only` \| `features_only` \| `remote_allowed` |

All contract constraints are **hard** constraints in V1.

```json
{
  "schema_version": "1.0",
  "contract_id": "CONTRACT_001",
  "task_id": "HUMAN_SEARCH_SEGMENTATION",
  "evidence_type": "instance_mask_set",
  "quality_metric": "target_f1",
  "quality_operator": ">=",
  "quality_threshold": 0.80,
  "deadline_s": 60.0,
  "communication_budget_mb": 50.0,
  "min_final_battery_frac": 0.20,
  "privacy_level": "remote_allowed"
}
```

The contract says *what* must be achieved. Task-specific matching rules — such as the
mask IoU at which a target counts as found — belong in the **task specification**, not
in every contract.

### 4.2 Task specification — *how the task is scored*

```json
{
  "schema_version": "1.0",
  "task_id": "HUMAN_SEARCH_SEGMENTATION",
  "evidence_type": "instance_mask_set",
  "target_type": "person",
  "ground_truth_type": "instance_masks_with_track_ids",
  "matching_rule": { "metric": "mask_iou", "operator": ">=", "threshold": 0.50 },
  "deduplication": { "method": "ground_truth_track_id" },
  "supported_quality_metrics": ["target_recall", "target_precision", "target_f1"]
}
```

A target is found when a predicted instance mask matches a ground-truth instance under
the matching rule. The evaluator supports target precision, target recall, and target
F1; the default example contract uses target F1. **Ground truth is hidden from the policy.**

### 4.3 Episode — *initial state*

```json
{
  "schema_version": "1.0",
  "episode_id": "EPISODE_001",
  "platform_id": "UAV_PLATFORM_001",
  "path_id": "PATH_001",
  "frame_stream_id": "STREAM_001",
  "initial_altitude_m": 40.0,
  "velocity_mps": 5.0,
  "initial_battery_frac": 0.80,
  "power_mode": "15W",
  "network_trace_id": "NETWORK_DEGRADING_001",
  "allowed_config_ids": ["CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG", "CFG_REMOTE_STRONG"],
  "initial_config_id": null,
  "seed": 42
}
```

The runner knows `network_trace_id`; **the policy never sees the future trace**, only the
current measurement.

### 4.4 Runtime state — *the policy's entire observation*

```json
{
  "current_time_s": 25.0,
  "frame_id": 25,
  "battery_frac": 0.55,
  "power_mode": "15W",
  "network": { "bandwidth_mbps": 8.0, "rtt_ms": 70.0, "packet_loss_frac": 0.01 },
  "current_config_id": "CFG_REMOTE_STRONG",
  "remaining_deadline_s": 35.0,
  "cumulative_energy_j": 5250.0,
  "cumulative_communication_mb": 19.5,
  "path_progress": 0.42,
  "evidence_summary": {
    "predicted_unique_targets": 3,
    "processed_frames": 20,
    "mean_prediction_confidence": 0.78
  }
}
```

`path_progress` is in [0, 1]. `current_config_id` is nullable. The evidence summary is
**prediction-derived only**: never true recall, never a hidden target count, never a
match outcome. See `docs/architecture.md` §5.

### 4.5 Configuration — *identity, not behaviour*

```json
{
  "config_id": "CFG_LOCAL_LIGHT",
  "model_id": "LIGHT_INSTANCE_SEG",
  "strategy": { "placement": "local", "precision": "int8", "parameters": {} }
}
```

Required V1 strategy fields: `placement` (`local` | `remote`), `precision`
(`fp32` | `fp16` | `int8`), optional `input_compression`, plus a controlled
`parameters` map for future strategy fields (pruning ratio, split point, input
resolution, frame sampling rate, early-exit index, accelerator, power mode).

**Power mode is not part of a V1 configuration** — it is fixed by the episode.

**`config_id` is an identifier only.** No component may infer behaviour by parsing it.

Configurations are shipped in a catalog document:

```json
{ "schema_version": "1.0", "catalog_id": "CATALOG_001", "configs": [ ... ] }
```

`catalog_id` is optional and identifies the pool; `configs` must be non-empty with unique
`config_id`s. A catalog is ordered by declaration so that iteration is deterministic, and
`ConfigCatalog.subset(ids)` produces the episode's allowed pool — the runner hands a policy
that subset, so a policy cannot select or even see a disallowed configuration.

### 4.6 Configuration profiles — *synthetic performance*, per platform

```json
{
  "CFG_LOCAL_LIGHT":   { "mean_latency_ms": 100.0, "mean_compute_energy_j": 3.0,  "upload_mb": 0.0, "download_mb": 0.0 },
  "CFG_LOCAL_STRONG":  { "mean_latency_ms": 450.0, "mean_compute_energy_j": 12.0, "upload_mb": 0.0, "download_mb": 0.0 },
  "CFG_REMOTE_STRONG": { "server_latency_ms": 100.0, "onboard_energy_j": 1.0, "upload_mb": 1.5, "download_mb": 0.05 }
}
```

Configuration *identity* is separate from *measured or simulated performance*. These
values are synthetic placeholders.

### 4.7 Platform profile

```json
{
  "schema_version": "1.0",
  "platform_id": "UAV_PLATFORM_001",
  "battery_capacity_wh": 100.0,
  "flight_power_w": 180.0,
  "communication_energy_j_per_mb": 0.5,
  "supported_power_modes": ["15W"]
}
```

### 4.8 Network trace

Deterministic, segment-based, with `bandwidth_mbps`, `rtt_ms`, `packet_loss_frac`.
At least three fixtures: **stable**, **degrading**, **disconnecting**.

```json
{
  "schema_version": "1.0",
  "trace_id": "NETWORK_DEGRADING_001",
  "segments": [
    { "start_s": 0,  "end_s": 20, "bandwidth_mbps": 20.0, "rtt_ms": 30.0,  "packet_loss_frac": 0.0 },
    { "start_s": 20, "end_s": 40, "bandwidth_mbps": 8.0,  "rtt_ms": 70.0,  "packet_loss_frac": 0.01 },
    { "start_s": 40, "end_s": 60, "bandwidth_mbps": 2.0,  "rtt_ms": 150.0, "packet_loss_frac": 0.03 }
  ]
}
```

Packet loss is recorded but needs no retransmission model in V1.

Segments are validated at load time to be **ordered and contiguous**, and each covers the
half-open interval `[start_s, end_s)`. A gap would make the observation at that time
undefined and an overlap would make it ambiguous; both are load errors rather than
lookup-time surprises, because the benchmark's determinism depends on every time in the
trace resolving to exactly one segment.

### 4.9 Path specification

The predefined path, reduced to the one property the loop consumes:

```json
{ "schema_version": "1.0", "path_id": "PATH_001", "length_m": 250.0, "description": "..." }
```

Waypoints, altitude profiles, and turn dynamics are deliberately absent — they would imply
a flight model V1 does not have, and nothing in the decision loop would read them. Combined
with the episode's fixed velocity, the length yields `path_progress`, which drives the
primary termination condition.

### 4.10 Privacy and transmitted payload

`local_only` and `remote_allowed` are decided by `strategy.placement` alone. `features_only`
needs to know what a remote configuration puts on the wire, which placement cannot express,
so a remote configuration declares it:

```json
"strategy": { "placement": "remote", "precision": "fp16",
              "parameters": { "transmitted_payload": "features" } }
```

Permitted values are `raw_input` and `features`. A remote configuration that does not declare
one is treated as transmitting **raw input** — the default fails closed, so forgetting the
parameter cannot quietly grant permission. Compression is not de-identification: a
`jpeg_q75` upload is still `raw_input`.

This lives in `parameters` rather than as a typed strategy field because only the privacy
rule reads it. It graduates to a typed field if the core starts reasoning about it more
broadly.

### 4.11 Shipped fixtures

| File | Contents |
|---|---|
| `data/contracts/contract_001.json` | The default example contract (`target_f1 >= 0.80`, `remote_allowed`) |
| `data/contracts/contract_002_local_only.json` | `local_only` privacy with a zero communication budget |
| `data/task_specs/human_search_segmentation.json` | The V1 task specification |
| `data/configs/config_catalog_001.json` | The three-configuration V1 catalog |
| `data/platforms/synthetic_uav_platform_001.json` | The example UAV platform |
| `data/network_traces/synthetic_network_{stable,degrading,disconnecting}_001.json` | The three required conditions |
| `data/paths/path_001.json` | The 250 m survey path (50 s at 5 m/s) |
| `data/episodes/episode_00{1,2,3}*.json` | One episode per network condition |

Files whose numbers are invented rather than chosen — platform profiles, network traces,
and later configuration profiles, predictions, and ground truth — are named `synthetic_*`
so a value lifted out of this repository cannot be mistaken for a measurement. A test
enforces the naming.

## 5. Policy action and validation

```python
class Policy(Protocol):
    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str: ...
```

An action is invalid if it is not a string at all, if the ID is unknown, if it is not in
`allowed_config_ids`, or if it violates the contract's privacy level. On an invalid action:

- record an invalid-action violation;
- if a valid current config exists, keep it;
- otherwise use the configurable safe fallback (default `CFG_LOCAL_LIGHT`);
- **never** crash the benchmark because of one invalid action.

The current configuration is re-checked before being kept: an episode's declared
`initial_config_id` never passed through validation, so it may itself be disallowed or
privacy-violating. The fallback is validated at construction — discovering mid-episode that
there is no legal action would leave the runner stuck.

A privacy-violating selection is **blocked, not executed and then penalised**. The simulator
must not model data leaving the vehicle in violation of its contract even hypothetically;
the attempt is recorded, and the episode's privacy constraint fails on the record.

## 6. Timing

Nominal one-frame-per-second stream. When inference exceeds one second:

- the UAV keeps following the predefined path;
- no second inference starts concurrently (V1: one active inference at a time);
- elapsed time advances by `max(1 s, inference_latency)`;
- the frame/path position advances accordingly, dropping intermediate frames.

The frame index is the wall clock floored to the frame interval: at t=1.9 s the frame
captured at 2.0 s does not exist yet. Frames that elapse during a long inference are never
seen — that dropped work is the cost being modelled, not an error, and the count is logged.

This must stay deterministic and covered by tests.

### Switching (V1 simplifications)

- The policy may select a config at every decision step.
- Selecting the **same** config is not a switch.
- Selecting a **different** config increments `configuration_switch_count`.
- All configs are assumed preloaded: **switching latency is zero, switching energy is zero.**

These are deliberate simplifications, not claims about real systems. No switching
penalty model is to be invented before it is specified.

## 7. Remote inference

```
remote_latency = upload_time + rtt + server_latency + download_time
upload_time_s  = 8 * upload_size_mb / bandwidth_mbps      (same relation for download)
```

If bandwidth is zero: the remote inference **fails**, produces no prediction, the failure
is recorded, a configurable timeout is applied, and the episode continues unless another
termination condition fires. A single failed remote inference never terminates an episode.

## 8. Battery

Battery is tracked internally as **energy in joules**.

```
capacity_j       = battery_capacity_wh * 3600
initial_energy_j = initial_battery_frac * capacity_j

step_energy_j = flight_power_w * elapsed_step_time_s
              + compute_or_onboard_energy_j
              + communication_energy_j

communication_energy_j = communication_volume_mb * communication_energy_j_per_mb

remaining_energy_j = max(0, previous_energy_j - step_energy_j)
battery_frac       = remaining_energy_j / capacity_j
```

Flight, compute, communication, and total energy are tracked **separately**. The model is
injected and replaceable (`BatteryModel` protocol); V1 ships `SimpleBatteryModel`.

## 9. Termination

Terminate when any of these becomes true:

- `path_progress >= 1.0`
- current time exceeds `deadline_s`
- battery reaches zero

On deadline exceedance: `deadline_success = false`, `mission_success = false`, and **all
partial evidence and resource logs are preserved**.

Conditions are evaluated in the order **path complete → deadline exceeded → battery
depleted**, and the first match is reported. The order is reporting precedence only: a step
that both finished the path and drained the battery did finish the path.

**The termination reason and constraint success are computed independently.** Constraint
success comes from final values, not from why the episode stopped, and the two can disagree.
Because the vehicle covers ground on wall-clock time, path completion lands on the path's own
duration regardless of what the policy selected — so a slow configuration does not fly
slower, it processes fewer frames. It can still miss the deadline: whichever step straddles
the path end carries the clock past it, so an episode can stop for `path_complete` at 64 s
against a 60 s deadline and fail `deadline_success`. `deadline_exceeded` itself fires only
when the path cannot be flown within the deadline at all.

A single failed remote inference is deliberately not a termination condition. Losing the
network is a situation the policy is meant to handle, not a reason to stop measuring how it
handles it.

## 10. Metrics

**Primary metric: Mission Success Rate.** An episode succeeds only if *all* applicable
hard constraints pass: quality, deadline, final battery, communication budget, privacy.

Per-episode: `mission_success`, quality value, `quality_success`, `deadline_success`,
`battery_constraint_success`, `communication_constraint_success`,
`privacy_constraint_success`, `final_battery_fraction`, `mission_completion_time_s`,
`mean_end_to_end_inference_latency_ms`, `total_communication_mb`, `total_energy_j`,
`flight_energy_j`, `compute_energy_j`, `communication_energy_j`,
`configuration_switch_count`, `constraint_violation_count`, violation margins,
invalid action count, failed inference count.

Aggregate: Mission Success Rate, Quality Success Rate, Deadline Success Rate, Battery
Constraint Success Rate, Communication Constraint Success Rate, Mean Final Battery
Fraction, Mean Mission Completion Time, Mean End-to-End Inference Latency, Mean
Communication Volume, Mean Configuration Switch Count, Mean Constraint Violation Count.

Clarifications:

- Final battery reserve is a **continuous** final battery value; Battery Constraint
  Success Rate is the **fraction of episodes** meeting the minimum final battery.
- Mission completion time and inference latency are **separate** metrics.
- `constraint_violation_count` counts **violated constraint categories**, not the number
  of time steps a single constraint stayed violated.
- Violation margins are reported: `deadline_violation_s`, `battery_violation_frac`,
  `communication_violation_mb`.

Mission-success logic consumes a standardised task result and must never name a
task-specific metric:

```json
{ "metric_name": "target_f1", "value": 0.84, "operator": ">=", "threshold": 0.80, "success": true, "details": {} }
```

### Adaptation latency — optional

Not a required V1 leaderboard metric: "appropriately adapted configuration" and its
stability criteria are not yet formally defined. The runner logs enough to compute it
later — network-change event timestamps, battery-threshold event timestamps, selected
config history, execution result history. A placeholder utility is permitted; it must not
block core V1.

## 11. Versioning and dependency policy

**Dependencies.** V1 declares **zero runtime dependencies**; `pytest` is the only dev
dependency. Schemas use stdlib `dataclasses` with centralised validation rather than a
validation library, so the benchmark runs anywhere Python 3.11+ runs. No heavy ML, ROS,
simulation, cloud, or GPU dependency may be added in this phase.

**Versioning.** Once fixture formats and interfaces are committed, avoid breaking changes
inside `develop/v1`. When a break is unavoidable: bump `schema_version`, update fixtures,
update tests, document the change — and never silently reinterpret an old field.
Migration support belongs in `aerointentbench/schemas/loading.py`; it is documented, not
implemented, in V1.

## 12. Explicit V1 assumptions

These are implementation choices, not universal truths. They live in configuration or
documented defaults, never as unexplained magic constants.

| Assumption | Rationale |
|---|---|
| One-second decision interval | Fixes the loop; nominal 1 fps stream |
| One active inference at a time | Avoids a concurrency model V1 does not need |
| Zero switching latency and energy | All configs assumed preloaded |
| Fixed power mode per episode | Power-mode selection is a future strategy dimension |
| Predefined path, no flight dynamics | The benchmark evaluates configuration choice, not control |
| Hard constraints only | Soft/weighted constraints are a future contract extension |
| Packet loss recorded, not modelled | No retransmission model is justified yet |

## 13. Milestone

V1 is done when: JSON inputs → deterministic 1-second decision loop → policy selects a
config → profile-based executor produces results → state and evidence update → episode
terminates → metrics JSON is generated. Nothing beyond that milestone is in scope.
