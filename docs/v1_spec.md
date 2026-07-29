# AeroIntentBench V1 specification

Status: **V1 complete and runnable.** This document is the normative reference. Where the
implementation and this document disagree, this document is the bug report.

> Every numeric value shown below is a **synthetic example**, not a measurement.

| | |
|---|---|
| §1–3 | What the benchmark evaluates, its scope, and the decision loop |
| §4 | Every schema, and the fixtures that instantiate them |
| §5–9 | Action validation, timing, remote inference, battery, termination |
| §10 | Metrics |
| §11–13 | Why the fixtures are sized as they are, what the baselines score, result files |
| §14–17 | Versioning, assumptions, known limitations, the milestone |

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
read as an intentional default. See §14 for where migrations would go.

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

#### Remote must not be a dominated option

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

### 4.7 Public profiles — *what a policy is told*

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

### 4.8 Platform profile

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

### 4.9 Network trace

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

### 4.10 Path specification

The predefined path, reduced to the one property the loop consumes:

```json
{ "schema_version": "1.0", "path_id": "PATH_001", "length_m": 4500.0, "description": "..." }
```

Waypoints, altitude profiles, and turn dynamics are deliberately absent — they would imply
a flight model V1 does not have, and nothing in the decision loop would read them. Combined
with the episode's fixed velocity, the length yields `path_progress`, which drives the
primary termination condition.

### 4.11 Privacy and transmitted payload

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

### 4.12 Ground truth and synthetic predictions

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

### 4.13 Two deduplications, deliberately different

Both the policy summary and the evaluator count "unique targets", and they count different
things:

- The **policy** sees distinct `predicted_target_id`s — what the system believes it found. A
  false positive inflates it, and the policy is given no way to discover that.
- The **evaluator** deduplicates by `ground_truth_track_id`, per the task specification.

Collapsing them would hand the policy its own true-positive count.

Scoring is per unique target, not per instance: a target counts as found if *any* prediction
matched it, and a predicted identity is a false positive if *none* of its instances matched
anything. Sixty sightings of one person are one find.

### 4.14 Measured discrimination

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

### 4.15 Empirical mask replay

§4.12 precomputes `mask_iou`: synthesis decides, per `(seed, frame_id, config_id, track_id)`,
whether a target is detected and what its overlap "would have been". **Empirical mask replay**
replaces that stand-in with real masks and a real IoU, without adding a model or a dataset —
the masks are recorded (here, synthetic), the way a hardware capture would eventually supply
them.

Which path scores an episode is decided by the **ground-truth form**, never by the executor.
A ground-truth file declaring per-frame `frames` of masks is scored empirically; one
declaring visibility `targets` (§4.12) is scored by the precomputed scalar. The two may sit
side by side in one `ground_truth/` directory; each file states which it is.

**Mask ground truth** — hidden, never policy-visible — carries one mask per target per frame.
A target keeps its `track_id` across frames; `ignore: true` marks a region excluded from the
target total and from penalising overlapping predictions.

```json
{
  "schema_version": "1.0",
  "frame_stream_id": "EXAMPLE_STREAM",
  "task_id": "HUMAN_SEARCH_SEGMENTATION",
  "frames": [
    { "frame_id": 0, "instances": [
        { "track_id": "GT_A", "category": "person",
          "mask": { "height": 8, "width": 8, "rows": ["00000000","01111000", "..."] } }
    ] }
  ]
}
```

**Empirical predictions** ride inside a normal replay record's opaque `prediction` payload,
so the replay schema itself is unchanged. Each instance carries `prediction_id`, `category`,
`confidence`, and a `mask`; it must **not** carry `ground_truth_track_id` (a prediction may
not name its answer) and any `mask_iou` on the wire is ignored, never trusted.

**Mask encoding.** One explicit, versioned form: `{ "height", "width", "rows" }`, one `'0'`/
`'1'` string per row. Dimensions are stated and recoverable; a ragged, mis-sized, or
bad-character bitmap is rejected. It decodes to a boolean pixel set — the canonical internal
mask. Other encodings (COCO RLE, polygons) would decode to the same type; none is needed in
V1. `mask_iou = |A ∩ B| / |A ∪ B|`; different dimensions raise rather than compare, and two
empty masks score `0.0` (no positive evidence of a target).

**Matching** is deterministic and one-to-one per frame: build the prediction × target IoU
matrix, keep pairs clearing the task spec's threshold (0.50), assign greedily by IoU
descending with identifier tie-breaks. Each prediction and each target pairs at most once.
Greedy rather than optimal (Hungarian) — V1 has no array or `scipy` dependency and the
per-frame instance counts are tiny; the limitation and its tie-breaking are tested.

**Scoring keeps two counting units strictly apart** — the fix that this metric semantics
depends on. Mixing them (unique-track TP over frame-level FP) produces a number in neither
unit, so the empirical path reports two families and never divides one into the other:

| Metric | Unit | Definition |
|---|---|---|
| `target_recall` (canonical mission metric) | mission, **track-level** | `unique_targets_found / total_unique_targets` — a track matched on any frame is one find, deduplicated by hidden `track_id` |
| `detection_precision` | frame, **detection-level** | `matched_detections / total_predictions` (non-ignored predictions) |
| `false_positive_detections` | frame | unmatched, non-absorbed detections (count) |
| `false_positives_per_processed_minute` | frame / examined footage | `false_positive_detections` per minute of *processed frames* at the nominal 1 fps — detection-error density of what was looked at (from the evaluator) |
| `false_positives_per_mission_minute` | frame / wall-clock | `false_positive_detections` per minute of *mission completion time* (added by the episode metrics, which know the clock); zero-duration ⇒ `0.0` |

**Track-level precision and F1 are not computed.** They require a prediction associated with
a persistent predicted *track* across frames; independent per-frame masks carry no such
identity (a per-frame `prediction_id` is not a track). They appear in the result as `null`
with `track_level_metrics_available: false` and a stated reason — never a mixed-unit number
under a track-level name. A contract may therefore score empirical missions on `target_recall`
(canonical) or `detection_precision`; asking for `target_precision`/`target_f1` fails loudly.
The example empirical contract uses `target_recall`. The precomputed path is unchanged and
still produces the track-level `QualityScores` (`target_precision`/`target_f1`).

**Provenance.** An empirically scored result is tagged `quality_evaluation:
"empirical_mask_iou"` in its quality details. With `executor_id`, this separates the three
sources a result can have: synthetic profile (`profile`, no tag), legacy scalar replay
(`replay`, no tag), and empirical mask replay (`replay`, tagged).

A tiny worked example ships under `data/examples/empirical_replay/` (8×8 masks, three tracked
people): one exact match, one partial match above threshold, one below-threshold detection,
one false positive, one missed target, one ignore region, and predictions under two configs
with differing latency and energy. It scores, by hand and in code, `target_recall 2/3`
(`unique_targets_found 2 / total 3`), `detection_precision 0.6` (`matched_detections 3 /
total_predictions 5`), `false_positive_detections 2`, `false_positives_per_processed_minute
40.0`, and `target_f1 null`. Run to termination through the CLI, the wall-clock
`false_positives_per_mission_minute` is `30.0` (the deadline admits one empty frame past the
last ground-truth frame, so 4.0 s of mission time), which is exactly why each rate is named
for its denominator.

### 4.16 Building empirical bundles from external data

§4.15 consumes a hand-authored bundle. The **bundle builder**
(`aerointentbench.tools.build_empirical_bundle`) is the bridge from *real* externally
produced data into that same format, so a researcher can run any model outside the benchmark
and convert its outputs without touching `EpisodeRunner`, `ReplayExecutor`, or the evaluator.
It executes no model and measures no hardware — it ingests, validates, and packages.

**Inputs.** A versioned manifest points at three source families (all stdlib-parseable, no
image-decoding dependency): a ground-truth JSON of per-frame mask instances; per-configuration
predictions as JSON Lines (`frame_id`, `prediction_id`, `category`, `confidence`, `mask` —
rejected if they carry a ground-truth track id or an on-the-wire `mask_iou`); and
per-configuration measurements as CSV (`frame_id, success, latency_s, compute_energy_j,
upload_mb, download_mb, failure_reason`). A blank required latency or energy cell is an error,
never a silent zero. The manifest also carries the mission scaffolding (platform, path,
network, contract — with defaults) and honest **provenance**: `data_origin`, and per config
`prediction_provenance` and `measurement_provenance`, so a hand-authored or estimated value is
never packaged as `measured`.

**Output.** A self-contained data root — configs, platform, profiles, path, network trace,
task spec, episode, contract, hidden mask ground truth, one replay record set, a frame
manifest, and a `provenance.json` — that the ordinary CLI runs with `--executor replay` and
no new flags. Policy-facing nominal profile costs are means of the supplied measurements,
marked as derived scaffolding, not a separate measurement.

**Coverage** is explicit. `strict` (the default) requires a measurement for every declared
`frame × config`; `sparse` converts a declared-but-missing pair into an explicit failed
replay record (`no_prediction_available`), matching `ReplayExecutor`'s own gap semantics. A
missing pair is never silently dropped.

**Determinism and safety.** Records are ordered by `(frame_id, config_id)`, ground truth by
`(frame_id, track_id)`, JSON written sorted; two builds of the same sources are byte-identical
but for one provenance `created_at` field (pinned with `--created-at`). Every generated file's
SHA-256 is recorded in the provenance manifest. Source paths resolve under the manifest's
directory and a `..` that escapes it is rejected. `validate_empirical_bundle` re-checks a
built bundle independently — schemas, config references, mask dimensions, no GT identity in
predictions, finite non-negative costs, coverage, and provenance counts and hashes against
what is on disk — accumulating every error rather than stopping at the first.

### 4.17 Shipped fixtures

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
| `data/examples/empirical_replay/` | A self-contained mini data root for §4.15: mask ground truth, a mask replay set, and its episode/contract. Synthetic correctness example, not a dataset |
| `data/examples/empirical_source/` | Source-format inputs for §4.16 (manifest, ground-truth JSON, prediction JSONL, measurement CSV) that the bundle builder converts into a runnable bundle. Synthetic; tests conversion correctness only |

Files whose numbers are invented rather than chosen — platform profiles, network traces,
and later configuration profiles, predictions, and ground truth — are named `synthetic_*`
so a value lifted out of this repository cannot be mistaken for a measurement. A test
enforces the naming. The empirical-replay example lives under its own `data/examples/`
root so it neither joins the shipped suite nor changes any pinned baseline.

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
- otherwise use the **resolved safe fallback**;
- **never** crash the benchmark because of one invalid action.

**Fallback resolution (model-agnostic).** The fallback is resolved by the composition root
*before* the episode starts (`benchmark.resolve_fallback_config_id`), not hard-coded to any
configuration name, so a data root built with its own configuration IDs is not obliged to
contain `CFG_LOCAL_LIGHT`. Precedence: **(1)** an explicit `--fallback-config-id` override;
**(2)** the episode's optional `fallback_config_id`; **(3)** the episode's `initial_config_id`
if usable; **(4)** the legacy `CFG_LOCAL_LIGHT` only when present and usable; **(5)** otherwise
fail before step 0, requiring an explicit fallback. A fallback *supplied* at (1) or (2) but
unusable is an error, not a fall-through. "Usable" means present in the catalog, in the allowed
pool, and permitted by the privacy level — never an arbitrary "first configuration in the
catalog". The resolved fallback is validated once more at `ActionValidator` construction.

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

## 7b. Execution backends

`--executor` selects how a chosen configuration becomes an `ExecutionResult`. The
`EpisodeRunner` depends only on the generic `Executor` interface; it never branches on the
backend. Construction goes through the executor registry, from a uniform `ExecutorContext`,
so the composition root — not the runner — owns backend-specific loading.

| Backend | Behaviour | Provenance |
|---|---|---|
| `profile` (default) | Costs synthesised from the per-platform profile plus the §7 remote-latency model. All values synthetic. | `profile` |
| `replay` | Serves precomputed `frame × config` records loaded from `data/predictions/`, resolved by the episode's `episode_id`. | `replay` |
| `real_segmentation` | V1 stub. **Construction raises**, so selection fails before step 0 with *"not implemented in V1. Use profile or replay."* | — |

**Replay resolution.** A record set declares the `episode_id` it was recorded from; a data
root holds at most one set per episode, checked at load. Selecting `replay` for an episode
with no set is an error naming how to record one — never a silent fall back to `profile`,
which would mix two provenances in one run. A missing individual record is a recorded failed
inference by default (a slow policy can skip frames), or an error under `--replay-strict`.

**Recording.** `python -m aerointentbench.tools.record_replay` captures a full-coverage set
from the profile executor. This is capture, not fabrication: it writes down exactly what the
profile run produced. It keeps only policy-visible prediction fields, so a replayed run
reproduces the profile run's resources exactly while quality scores as all-false-positive —
replay is exact for the plumbing, and a set that retained the hidden fields would be needed
to reproduce a quality score. When real models arrive, a hardware capture emits this same
format and nothing downstream changes.

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

## 11. Mission scale, and what each constraint actually binds

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

## 12. Baseline policies and measured difficulty

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

Over the three shipped episodes under `contract_001` (`target_f1 >= 0.85`, 400 MB, 20 %
reserve), pooled over 50 seeds per episode (**n = 150**):

| Policy | MSR | 95 % CI | mean F1 | quality | comms |
|---|---:|---|---:|---:|---:|
| `always_local_light` | **0.7 %** | [0.1, 3.7] | 0.682 | 1 % | 100 % |
| `always_local_strong` | **64.7 %** | [56.7, 71.9] | 0.862 | 65 % | 100 % |
| `always_remote_strong` | **0.0 %** | [0.0, 2.5] | 0.878 | 64 % | **0 %** |
| `rule_based` | **78.0 %** | [70.7, 83.9] | 0.883 | 78 % | 100 % |
| `rule_based` (profiles hidden) | **0.7 %** | [0.1, 3.7] | 0.682 | 1 % | 100 % |

`rule_based` against the best static baseline: **+13.3 points, z = 2.58, p = 0.010** —
significant at 95 %.

Each baseline fails for its own reason, which is what makes the suite informative:
`local_light` almost never reaches the quality threshold; `remote_strong` scores well on
quality and busts the communication budget **every single time**; `local_strong` is
genuinely competitive and loses where the network was good enough that remote was worth
spending on.

#### Why these are pooled, and why the earlier numbers were wrong

An earlier revision of this document reported 0 % / 67 % / 0 % / 100 % from **three
episodes**, and claimed adaptation beat every static baseline. That claim was not supported.

Over three episodes a success rate can only be 0, 1/3, 2/3 or 1, and 2/3 carries a 95 %
interval of **[20.8 %, 93.9 %]** — seventy-three points wide. Meanwhile quality scores vary
by roughly 0.05 between seeds while the margins deciding pass/fail are around 0.02. Three
samples cannot resolve that.

Pooled at the *old* 0.80 threshold the same comparison was 96.0 % against 92.0 %:
**z = 1.46, p = 0.14, not significant**. The threshold was raised to 0.85 precisely because
at 0.80 the strong local baseline sits near the ceiling and leaves no room to distinguish
anything:

| threshold | light | local_strong | remote | rule_based | difference | z | significant |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 8.0 % | 92.0 % | 0.0 % | 96.0 % | +4.0 | 1.46 | no |
| 0.83 | 0.0 % | 74.0 % | 0.0 % | 90.0 % | +16.0 | 3.69 | yes |
| **0.85** | **0.0 %** | **64.7 %** | **0.0 %** | **78.0 %** | **+13.3** | **2.58** | **yes** |
| 0.87 | 0.0 % | 50.0 % | 0.0 % | 66.0 % | +16.0 | 2.85 | yes |
| 0.90 | 0.0 % | 26.0 % | 0.0 % | 38.0 % | +12.0 | 2.25 | yes |

0.85 keeps the effect significant while leaving `rule_based` at 78 % — headroom for a better
policy to occupy, which 96 % would not have left.

**Report Mission Success Rate with `--repeats`.** The default of one repeat runs the
episodes as written and is right for reproducing a specific run; it is not enough to compare
two policies. `aggregate.mission_success_ci_95` is emitted with every result, and the CLI
warns when a suite is smaller than thirty runs.

#### The one rate that is not a sampling artifact

`always_remote_strong` measures exactly 0.0 % at n = 150, CI [0.0 %, 2.5 %]. That is
**structural, not noise**. Communication volume is deterministic — 1.55 MB per processed
frame — so the 400 MB budget is exceeded on every run, by 449 to 995 MB. Its quality is
second-best of all the baselines. More episodes will never move it.

That distinction matters when reading any 0 % in this benchmark: it may mean "too few
samples to see a rare success", or it may mean "this cannot happen". Here the intervals
separate the two.

### A fixture invariant this exposed

`EPISODE_003` originally allowed only `CFG_LOCAL_LIGHT` and `CFG_REMOTE_STRONG`, and **no
policy could pass it** — light never reached the threshold and remote could not stay inside
the budget. An episode nothing can pass measures nothing, exactly as an episode doomed on
battery by flight alone would (§11). `CFG_LOCAL_STRONG` was added to its pool, after which
`rule_based` (0.872) beats `always_local_strong` (0.842) there.

## 13. Result files

A result file carries `schema_version` like every other document:

```json
{
  "schema_version": "1.0",
  "benchmark_version": "0.1.0.dev0",
  "policy": "rule_based",
  "executor_id": "profile",
  "repeats": 1,
  "aggregate": { "episode_count": 3, "mission_success_rate": 1.0, "rates": {...}, "means": {...} },
  "episodes": [ { "mission_success": true, "executor_id": "profile", "quality": {...},
                  "constraints": {...}, "violations": {...}, "resources": {...},
                  "behaviour": {...}, "record": {...} } ]
}
```

Written with sorted keys, so re-running an unchanged benchmark produces a **byte-identical
file** and a determinism regression shows up as a diff in review.

`executor_id` records which backend produced the result — `profile`, `replay`, or an
`unregistered:<ClassName>` for a programmatically injected one. It is read from the executor
object rather than named alongside it, so it can never misreport the backend that ran; the
suite-level `executor_id` is the sorted set of what actually ran across its episodes.

**The step log and raw evidence are excluded by default.** Beyond size — a 900 s episode
logs 900 steps and thousands of predicted instances — raw evidence carries
`ground_truth_track_id` and `mask_iou` on every instance, so a shared result file would
otherwise publish the answers. `--include-detail` opts in; a detailed file must not be
published.

The record retains an `adaptation_log` that **no V1 metric consumes**: network-change
timestamps, battery-threshold crossings, and the full configuration selection history. It is
kept so adaptation latency can be defined and computed later without rerunning a campaign.

## 14. Versioning and dependency policy

**Dependencies.** V1 declares **zero runtime dependencies**; `pytest` is the only dev
dependency. Schemas use stdlib `dataclasses` with centralised validation rather than a
validation library, so the benchmark runs anywhere Python 3.11+ runs. No heavy ML, ROS,
simulation, cloud, or GPU dependency may be added in this phase.

**Versioning.** Once fixture formats and interfaces are committed, avoid breaking changes
inside `develop/v1`. When a break is unavoidable: bump `schema_version`, update fixtures,
update tests, document the change — and never silently reinterpret an old field.
Migration support belongs in `aerointentbench/schemas/loading.py`; it is documented, not
implemented, in V1.

### 14.1 Frozen empirical schema registry

Every externally consumed format below is at **`schema_version` `"1.0"`** and frozen for V1.
All are loaded through `aerointentbench/schemas/loading.py`, which **rejects any version other
than `"1.0"`** (`SchemaVersionError`) rather than reinterpreting it, validates fields strictly
(unknown keys are errors), and serialises deterministically (sorted keys). The empirical
execution modes are distinguished as: **synthetic profile** (`executor_id="profile"`, no
`quality_evaluation` tag), **legacy scalar replay** (`executor_id="replay"`, no tag),
**empirical mask replay** (`executor_id="replay"`, `quality_evaluation="empirical_mask_iou"`).

| Schema | Version | Producer → Consumer | Hidden / private | Notes |
|---|---|---|---|---|
| Empirical mask **ground truth** (`frames[]` of masks) | 1.0 | dataset/bundle → evaluator | **whole file** (masks, track ids); never policy-visible | §4.15; discriminated from interval GT by the `frames` key |
| Empirical **prediction** payload (inside a replay record) | 1.0 | model/bundle → evaluator | none — masks are prediction output, but must **not** carry `ground_truth_track_id` or a trusted `mask_iou` | §4.15 |
| **Replay record set** (`data/predictions/*.json`) | 1.0 | recorder/bundle → `ReplayExecutor` | none | keyed by `episode_id`; at most one per episode |
| Bundle-builder **source manifest** | 1.0 | researcher → `build_empirical_bundle` | none | the versioned envelope for its prediction JSONL / measurement CSV sources (those inherit its version) |
| Bundle **provenance manifest** (`provenance.json`) | 1.0 | `build_empirical_bundle` → readers/validator | none | records `data_origin`, per-config `prediction_provenance` / `measurement_provenance`, SHA-256 of every generated file |
| Pilot intermediate (GT JSON / prediction JSONL / measurement CSV) | 1.0 (GT & manifest) | pilot runner → bundle builder | GT hidden | JSONL/CSV are per-config streams under the manifest's version; `experiments/real_segmentation_pilot/` |

**Compatibility guarantee for V1:** these field sets and their meanings are stable within
`develop/v1`. `initial_config_id` and the new optional `fallback_config_id` on an episode are
additive and backward-compatible (absent ⇒ prior behaviour). A future breaking change bumps
the version and adds a migration in `loading.py`; an unsupported future version fails loudly
today rather than being guessed at.

## 15. Explicit V1 assumptions

These are implementation choices, not universal truths. They live in configuration or
documented defaults, never as unexplained magic constants.

| Assumption | Rationale |
|---|---|
| One-second decision interval | Fixes the loop; nominal 1 fps stream |
| One active inference at a time | Avoids a concurrency model V1 does not need |
| Zero switching latency and energy | All configs assumed preloaded |
| Fixed power mode per episode | Power-mode selection is a future strategy dimension |
| Predefined path, no flight dynamics | The benchmark evaluates configuration choice, not control |
| 900 s mission, battery reserve as a guard | Flight power dominates compute by 15–60×, so battery cannot discriminate; the length exists to make it a *live observation* (§11) |
| Hard constraints only | Soft/weighted constraints are a future contract extension |
| Packet loss recorded, not modelled | No retransmission model is justified yet |

## 16. Known limitations

Recorded rather than hidden. None blocks the V1 milestone; each is a candidate for the next
version.

| Limitation | Detail |
|---|---|
| **`initial_altitude_m` has no consumer** | The episode schema carries it and nothing reads it. In a realistic benchmark altitude would drive ground sample distance and therefore detection difficulty — 40 m and 100 m give very different mask sizes. Today it is decorative, and should either be wired into the synthetic prediction model or removed. |
| **The quality score is quantised** | Twenty ground-truth targets means recall moves in steps of 0.05. Adequate for separating adaptive from static policies; too coarse to rank policies finely. More targets would smooth it. |
| **Three episodes are not a sample** | A success rate over the shipped suite can only be 0, 1/3, 2/3 or 1, and 2/3 carries a 95 % interval seventy-three points wide. `--repeats` pools seeds to narrow it, but the three episodes share one ground-truth stream, so they remain correlated. Additional streams would do more than more seeds. |
| **The battery reserve does not discriminate** | Flight power dominates on-board compute by 15–60×, so no policy can move the final battery fraction much. It is a live observation and a guard, not a scoring axis (§11). |
| **Adaptation latency is unimplemented** | "Appropriately adapted" is not formally defined, so no metric is computed. The data to compute one later is logged (§13). |
| **Switching is free** | Zero latency and zero energy, all configurations assumed preloaded. Real model swapping costs both. |
| **Packet loss is inert** | Recorded in the step log; no retransmission or corruption model. |
| **One catalog per data root** | Selecting among several configuration catalogs is not specified. |
| **`features_only` is untested end to end** | The rule is implemented and unit-tested, but no shipped episode exercises it, because no shipped configuration declares a feature payload. |
| **Empirical masks are still synthetic** | §4.15 computes real IoU from real masks and matches one-to-one, but the masks it consumes are a hand-authored correctness example, not model output. No real segmentation model, dataset, or `frame × config` capture is included; `real_segmentation` remains a construction-time stub. Empirical replay is the seam such a capture plugs into. |
| **No empirical track-level precision or F1** | Track-level precision and F1 need a prediction tied to a persistent predicted *track* across frames, and V1 empirical replay carries independent per-frame masks with no such identity. Rather than mix a track count with a detection count, empirical scoring reports `target_recall` (track-level) and `detection_precision` (detection-level) separately, and leaves `target_f1` `null` (§4.15). Persistent predicted-track association — a tracker over the predictions — would be needed to add them. |
| **Greedy matching, not optimal** | Empirical matching is greedy by IoU, not the globally optimal assignment. The two disagree only under contrived overlaps at the per-frame instance counts V1 sees; adding an optimal matcher would mean an array dependency V1 forbids. |

## 17. Milestone

V1 is done when: JSON inputs → deterministic 1-second decision loop → policy selects a
config → profile-based executor produces results → state and evidence update → episode
terminates → metrics JSON is generated. Nothing beyond that milestone is in scope.

**Reached.** `python -m aerointentbench.run_benchmark --suite --contract
data/contracts/contract_001.json --policy rule_based` runs three 900-step episodes and
writes a metrics file, on CPU, with no runtime dependencies. Re-running produces a
byte-identical result.

The suite is pinned in `tests/reference/baseline_results.json`: every baseline's score on
the shipped fixtures, compared on every test run. A change to a fixture, a profile, the
prediction model, the evaluator, or a policy surfaces there as a diff, so the benchmark
cannot move without someone approving the move.
