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
  "deadline_s": 960.0,
  "communication_budget_mb": 400.0,
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
  "current_time_s": 400.0,
  "frame_id": 400,
  "battery_frac": 0.55,
  "power_mode": "15W",
  "network": { "bandwidth_mbps": 8.0, "rtt_ms": 70.0, "packet_loss_frac": 0.01 },
  "current_config_id": "CFG_REMOTE_STRONG",
  "remaining_deadline_s": 560.0,
  "cumulative_energy_j": 73200.0,
  "cumulative_communication_mb": 186.0,
  "path_progress": 0.44,
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

The shipped schema uses one record shape for local and remote alike:

```json
{
  "schema_version": "1.0",
  "platform_id": "UAV_PLATFORM_001",
  "profiles": {
    "CFG_LOCAL_LIGHT":   { "compute_latency_ms": 100.0, "onboard_energy_j":  3.0, "upload_mb": 0.0, "download_mb": 0.00, "quality_tier": "low" },
    "CFG_LOCAL_STRONG":  { "compute_latency_ms": 450.0, "onboard_energy_j": 12.0, "upload_mb": 0.0, "download_mb": 0.00, "quality_tier": "medium" },
    "CFG_REMOTE_STRONG": { "compute_latency_ms": 100.0, "onboard_energy_j":  1.0, "upload_mb": 1.5, "download_mb": 0.05, "quality_tier": "high" }
  }
}
```

`compute_latency_ms` is time spent computing *wherever that happens* — on-vehicle inference
for a local configuration, server-side inference for a remote one. It is **not** a remote
configuration's end-to-end latency: transfer and RTT depend on the network at the moment of
execution, so the executor adds them. `onboard_energy_j` is energy drawn from the vehicle
battery: full inference locally, the smaller capture-and-encode cost remotely.

### Remote must not be a dominated option

`CFG_REMOTE_STRONG` runs `SERVER_XL_INSTANCE_SEG` — a model the vehicle cannot host — and
carries the **highest** quality tier. This matters for the benchmark to have a trade-off at
all: remote costs bandwidth and fails outright when the link drops, so if it were merely
equal in quality to a local option no policy would ever rationally select it, and the
communication budget would degenerate from a trade-off into a tax. A test enforces that
remote out-ranks every local option.

The resulting tension, using the shipped profiles:

| Configuration | Quality | 20 Mbps | 8 Mbps | 2 Mbps | Disconnected |
|---|---|---:|---:|---:|---|
| `CFG_LOCAL_LIGHT` | low | 0.10 s | 0.10 s | 0.10 s | 0.10 s |
| `CFG_LOCAL_STRONG` | medium | 0.45 s | 0.45 s | 0.45 s | 0.45 s |
| `CFG_REMOTE_STRONG` | **high** | 0.75 s | 1.72 s | **6.45 s** | **fails** |

Remote is the best option while the link is good and becomes ruinous as it degrades — at
2 Mbps one inference spans more than six frame intervals. Deciding when to give it up is the
adaptation the benchmark measures.

### 4.6b Public profiles: what a policy is told

Policies cannot infer quality from `config_id` — the architecture forbids parsing it, and
`model_id` is equally off-limits. Without some channel, a policy could not reason about the
quality constraint at all. The channel is an explicit, run-level setting:

```python
PublicProfile(config_id=..., expected_latency_ms=100.0, expected_upload_mb=1.5, quality_tier="high")
```

- **`quality_tier` is ordinal** (`low` / `medium` / `high`, with a `rank`), never a number.
  A policy needs to know one configuration is more accurate than another; it must not get a
  figure close enough to the true score to plan against.
- **`expected_latency_ms` is compute time only.** A policy estimates a remote
  configuration's end-to-end latency from the bandwidth it already observes, so it gains no
  foresight about a network it has not yet seen.
- **Exact energy is not disclosed.**

`PublicProfileView.hidden()` supplies the profile-blind mode. A policy must treat an empty
view as legitimate — whether profiles are public is a property of the run, and a policy that
crashes without them is not a valid submission.

The middle ground is deliberate. Exact profiles would turn a benchmark about adapting under
uncertainty into an offline planning exercise; nothing at all would make the quality
constraint unreasonable-about.

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
    { "start_s": 0,   "end_s": 320, "bandwidth_mbps": 20.0, "rtt_ms": 30.0,  "packet_loss_frac": 0.0 },
    { "start_s": 320, "end_s": 640, "bandwidth_mbps": 8.0,  "rtt_ms": 70.0,  "packet_loss_frac": 0.01 },
    { "start_s": 640, "end_s": 960, "bandwidth_mbps": 2.0,  "rtt_ms": 150.0, "packet_loss_frac": 0.03 }
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
{ "schema_version": "1.0", "path_id": "PATH_001", "length_m": 4500.0, "description": "..." }
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

### 4.11 Ground truth and synthetic predictions

Ground truth is keyed on the **frame stream**, not the episode: several episodes may fly the
same scene under different battery, network, or contract conditions and must be scored
against the same answers. Each target is a track ID plus an inclusive visibility interval.

```json
{
  "schema_version": "1.0",
  "frame_stream_id": "STREAM_001",
  "task_id": "HUMAN_SEARCH_SEGMENTATION",
  "targets": [
    { "track_id": "GT_L1",  "first_frame_id": 40,  "last_frame_id": 95 },
    { "track_id": "GT_F01", "first_frame_id": 110, "last_frame_id": 110 }
  ]
}
```

**Visibility duration is what makes latency cost quality.** A target in view for one frame
is missed outright by a configuration slow enough to skip it — at 2 Mbps a remote inference
spans more than six frame intervals. Without fleeting targets every configuration would find
everything and the benchmark would measure nothing, so the shipped stream carries 20 targets:
4 sustained (~55 frames), 6 brief (~9), and 10 fleeting (1–2), spread across all three
network segments.

A predicted instance carries two kinds of field:

| Field | Visibility |
|---|---|
| `prediction_id`, `frame_id`, `predicted_target_id`, `confidence` | aggregated into the policy summary |
| `ground_truth_track_id`, `mask_iou` | **hidden**; evaluator only |

The hidden pair stands in for real mask comparison — a production system would store a mask
and compute IoU. The evaluator reads `mask_iou` through the task spec's matching rule, so
replacing synthesis with real masks changes what fills the field, not what reads it.

Synthesis is deterministic in `(seed, frame_id, config_id, track_id)`, hashed with
`hashlib` rather than the builtin `hash`, whose string seed is randomised per process.
Behaviour follows the configuration's `quality_tier`:

| Tier | Detection probability | Mask IoU range | False positives / frame |
|---|---:|---|---:|
| low | 0.55 | 0.30 – 0.75 | 0.004 |
| medium | 0.75 | 0.42 – 0.88 | 0.002 |
| high | 0.92 | 0.48 – 0.96 | 0.001 |

The IoU range **straddles the 0.50 matching threshold** on purpose: a weak configuration can
see a person and still fail to segment them well enough to count. Detecting and matching are
separate events.

### 4.11b Two deduplications, deliberately different

Both the policy summary and the evaluator count "unique targets", and they count different
things:

- The **policy** sees distinct `predicted_target_id`s — what the system believes it found. A
  false positive inflates it, and the policy is given no way to discover that.
- The **evaluator** deduplicates by `ground_truth_track_id`, per the task specification.

Collapsing them would hand the policy its own true-positive count.

Scoring is per unique target, not per instance: a target counts as found if *any* prediction
matched it, and a predicted identity is a false positive if *none* of its instances matched
anything. Sixty sightings of one person are one find.

### 4.11c Measured discrimination (synthetic fixtures)

Running the full 900 s `EPISODE_001` under static strategies:

| Strategy | Frames | Found | Recall | Precision | F1 | Comms |
|---|---:|---:|---:|---:|---:|---:|
| always local-light (low) | 900 | 15/20 | 0.750 | 0.625 | 0.682 | 0 MB |
| always local-strong (medium) | 900 | 16/20 | 0.800 | 0.941 | 0.865 | 0 MB |
| always remote (high) | 548 | 18/20 | 0.900 | 0.947 | 0.923 | **849 MB — over budget** |
| budget-rationed remote | 900 | 17/20 | 0.850 | 0.895 | 0.872 | 400 MB |

Two properties this confirms, and one caveat:

- **Quality tier moves recall**, which is the intended signal. An earlier fixture with longer
  visibility windows saturated recall at 1.000 for every configuration, leaving false-positive
  count as the only discriminator — which perversely rewarded processing *fewer* frames.
- **All-remote is stopped by the communication budget**, not by quality. It scores best and
  cannot afford to.
- **Caveat: the score is quantised.** Twenty targets means recall moves in steps of 0.05, so
  small differences between policies are not meaningful. That is adequate for V1's coarse
  comparisons and would need more targets for fine ones.

The current `contract_001` threshold of 0.80 is reachable by `always local-strong` without any
adaptation. Whether to raise it so that only adaptive policies pass is a **calibration decision
deferred to `feature/v1-policies`**, when real policies exist to calibrate against; tuning
difficulty against hand-written strategy stubs would be fitting the benchmark to its own probe.

### 4.12 Shipped fixtures

| File | Contents |
|---|---|
| `data/contracts/contract_001.json` | The default example contract (`target_f1 >= 0.80`, `remote_allowed`) |
| `data/contracts/contract_002_local_only.json` | `local_only` privacy with a zero communication budget |
| `data/task_specs/human_search_segmentation.json` | The V1 task specification |
| `data/configs/config_catalog_001.json` | The three-configuration V1 catalog |
| `data/platforms/synthetic_uav_platform_001.json` | The example UAV platform |
| `data/network_traces/synthetic_network_{stable,degrading,disconnecting}_001.json` | The three required conditions |
| `data/paths/path_001.json` | The 4500 m survey path (900 s at 5 m/s) |
| `data/profiles/synthetic_profiles_uav_platform_001.json` | Per-configuration costs and quality tiers |
| `data/predictions/synthetic_replay_example.json` | A four-record replay set exercising the replay backend |
| `data/ground_truth/synthetic_human_search_stream_001.json` | 20 targets over STREAM_001: 4 sustained, 6 brief, 10 fleeting |
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
is recorded, a configurable timeout is applied (default 2 s), and the episode continues
unless another termination condition fires. A single failed remote inference never
terminates an episode.

A failed remote execution still charges `onboard_energy_j` — the vehicle captured and
encoded the frame before discovering it could not send it — but counts **zero**
communication, since nothing reached the server. Packet loss is recorded in the step log
and applies no latency penalty: V1 models no retransmission.

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

## 10b. Mission scale, and what each constraint actually binds

The V1 mission is **900 s (15 min) over 4500 m at 5 m/s**, against a 960 s deadline, a
400 MB communication budget, and a 20 % battery reserve. Those magnitudes are chosen, not
arbitrary, and the reasoning is recorded here because it determines what the benchmark can
and cannot measure.

### Flight power dominates on-board compute

On the example platform, per second of mission:

| Configuration | Total | Flight | Compute + communication |
|---|---:|---:|---:|
| `CFG_LOCAL_LIGHT` | 183.00 J/s | 98.4 % | 1.6 % |
| `CFG_LOCAL_STRONG` | 192.00 J/s | 93.8 % | 6.3 % |
| `CFG_REMOTE_STRONG` | 181.78 J/s | 99.0 % | 1.0 % |

The spread a policy can produce is **5.6 % of the consumption rate**, and it is a *ratio* —
lengthening the mission scales flight and compute together and does not widen it. This is
physical, not a modelling shortcut: a UAV draws hundreds of watts to stay airborne while a
15 W accelerator spends single-digit joules per frame.

**Consequence: the battery reserve is a guard, not a discriminating constraint.** Making it
bind would require placing the mission inside a ~4.7 % window (1125–1180 s) where all-light
passes and all-strong fails, which any change to the profile numbers would invalidate. That
knife-edge was rejected.

### What the mission length is actually for

The battery is a **policy observation**, and an observation that does not move is useless.
At the original 50 s scale the battery went 0.800 → 0.775 — a rule such as *"below 30 %,
switch to the light configuration"* could never fire, so the benchmark could not distinguish
a battery-aware policy from a battery-blind one. At 900 s it goes **0.800 → ~0.34**, visibly
approaching the 0.20 reserve. A guard test enforces this span.

### What each hard constraint does

| Constraint | Status |
|---|---|
| **Quality** | **Binds.** The central trade-off: latency → dropped frames → missed targets. |
| **Communication** | **Binds.** All-remote moves 1395 MB against a 400 MB budget, so remote is affordable for ~29 % of frames and must be rationed. |
| **Deadline** | Weak. Path completion lands on the path's own duration; the deadline is missed only when the step straddling the path end overshoots it. |
| **Battery** | **Guard, not discriminator.** Passes under every fixture and configuration; it moves enough to condition behaviour, but does not decide outcomes. |
| **Privacy** | Binds where the contract restricts it (`contract_002_local_only`); blocked at validation and recorded. |

Two fixture invariants follow, both enforced by tests: no episode may be doomed on battery
by flight alone (flight energy is not policy-controllable, so it must never decide an
outcome), and the path must be flyable inside every deadline (path completion has to be a
reachable outcome).

## 10c. Baseline policies and measured difficulty

V1 ships four baselines. None is claimed to be optimal; they exist so that a learned or
optimisation-based policy has something meaningful to beat.

- **`always_local_light` / `always_local_strong` / `always_remote_strong`** — `StaticPolicy`
  with a fixed configuration ID. Naming a configuration is not the same as parsing an ID to
  deduce behaviour; "always use X" *is* the policy. If X is outside an episode's allowed
  pool the policy still returns it, and the recorded invalid-action count is the honest
  report that the baseline does not apply there.
- **`rule_based`** — filters by privacy, then reachability, then communication budget; takes
  the fastest option under deadline or battery pressure; drops anything over a latency
  ceiling; and finally takes the best quality tier that survives.

### Measured Mission Success Rate

Across the three shipped episodes under `contract_001` (`target_f1 >= 0.80`, 400 MB, 20 %
reserve):

| Policy | EP1 | EP2 | EP3 | MSR |
|---|---|---|---|---:|
| `always_local_light` | fail | fail | fail | **0 %** |
| `always_local_strong` | pass | fail | pass | **67 %** |
| `always_remote_strong` | fail | fail | fail | **0 %** |
| `rule_based` | pass | pass | pass | **100 %** |
| `rule_based` (profiles hidden) | fail | fail | fail | **0 %** |

Each baseline fails for its own reason, which is what makes the suite informative:
`local_light` never reaches the quality threshold; `remote_strong` scores highest of all
(F1 0.923–0.950) and busts the communication budget every time; `local_strong` is genuinely
competitive and loses only where the network is good enough that remote was worth spending
on. The rule-based policy passes by using remote while bandwidth is high, then rationing.

**Calibration outcome: the 0.80 threshold stands.** It was an open question whether static
policies passed too easily — on `EPISODE_001` alone, `always_local_strong` (0.865) clears it
without adapting. Across the suite it does not, so no adjustment was made. Tuning the
threshold against hand-written probes would have been fitting the benchmark to its own test.

Two caveats:

- **`rule_based` at 100 % leaves no headroom.** With three episodes a competent heuristic can
  sweep. A larger episode suite is needed before the metric can rank policies rather than
  merely separate adaptive from static.
- **Profile-blind runs collapse to 0 %.** Without a quality tier a policy genuinely cannot
  tell configurations apart, so it falls back to catalog order. That is the honest cost of
  hiding profiles, and the measured justification for disclosing an ordinal tier (§4.6b).

### A fixture invariant this exposed

`EPISODE_003` originally allowed only `CFG_LOCAL_LIGHT` and `CFG_REMOTE_STRONG`, and **no
policy could pass it** — light never reached the threshold and remote could not stay inside
the budget. An episode nothing can pass measures nothing, exactly as an episode doomed on
battery by flight alone would (§10b). `CFG_LOCAL_STRONG` was added to its pool, after which
`rule_based` (0.872) beats `always_local_strong` (0.842) there.

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
| 900 s mission, battery reserve as a guard | Flight power dominates compute by 15–60×, so battery cannot discriminate; the length exists to make it a *live observation* (§10b) |
| Hard constraints only | Soft/weighted constraints are a future contract extension |
| Packet loss recorded, not modelled | No retransmission model is justified yet |

## 13. Milestone

V1 is done when: JSON inputs → deterministic 1-second decision loop → policy selects a
config → profile-based executor produces results → state and evidence update → episode
terminates → metrics JSON is generated. Nothing beyond that milestone is in scope.
