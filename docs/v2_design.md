# AeroIntentBench V2 — the minimal visual closed loop

V2 inserts a real visual world between the mission state and the perception executor.
It is the deliberate bridge between the abstract V1 benchmark and a future physical
(V3) simulation — **not** a realistic UAV simulator, and **not** a source of real-world
perception performance numbers.

## 1. Version progression

| | V1 (frozen) | V2 (this) | Future V3 |
|---|---|---|---|
| World | abstract frame counter | large 2D aerial raster | high-fidelity 3D environment |
| UAV | path progress scalar | predefined 2D trajectory, fixed altitude/speed | flight dynamics |
| Observation | none (replayed records) | position-dependent image crop | rendered/real camera |
| Executor | profile/replay records | lightweight image-based models | real perception models & strategies |
| Latency | profile value advances a clock | configured latency moves the UAV and skips observations | measured on hardware |
| Energy/network | synthetic formulas | configured per call (simulated) | hardware & network measurements |
| Validates | benchmark logic, schemas, metrics | architecture, interfaces, closed-loop semantics, policy trade-offs | physical mission evaluation |

V1 flow: `contract + state → policy → replayed profile result → mathematical update →
evaluation`. V1 remains fully functional — `aerointentbench.run_benchmark` and every V1
schema, fixture, and test are untouched.

V2 flow (authoritative):

```
2D aerial world → predefined UAV trajectory → position-dependent image crop
→ configuration-selection policy → image-based executor → prediction + latency
→ movement + mission-state update → next position-dependent observation
```

## 2. What V2 answers

- Observations change as the UAV moves (the crop is computed from `world image + UAV
  position + camera footprint` at render time — never a directory of pre-generated
  frames).
- A policy selects between model-strategies mid-mission on the **unchanged V1 policy
  interface**.
- The executor processes actual image content; its prediction depends on RGB.
- A slower model skips later observations; a target can be missed because the UAV
  moved during inference.
- Accuracy / latency / battery / deadline / communication trade-offs surface in
  mission-level metrics with the frozen V1 empirical vocabulary.
- The 2D simulator sits behind narrow interfaces (`WorldSource`, `Trajectory`,
  `CameraRenderer`, `ImageExecutor`) so a 3D backend can replace it without
  redesigning the benchmark.

## 3. Architecture

```
aerointentbench/v2/          (optional subpackage; extras: aerointentbench[v2])
  scenario.py    V2Scenario schema (scenario_schema_version "2.0", strict validation)
  world.py       WorldSource backends: RasterioWorld (GeoTIFF, windowed reads),
                 ImageWorld (PNG/JPEG), ArrayWorld (tests)
  trajectory.py  PolylineTrajectory + lawnmower generator; position_at(t)
  objects.py     ObjectLayer: synthetic targets/distractors, semantic+instance GT
  camera.py      CameraRenderer → Observation (RGB + evaluator-only GT)
  executors.py   fast_weak / slow_strong image executors + extension point
  evaluation.py  MissionEvaluator (V1 empirical metric semantics)
  runner.py      MissionRunner: the canonical closed loop
  visualize.py   overview + observation debug panels (headless PNG)
  cli.py         validate / overview / run
```

**Dependency boundary.** The V1 core keeps zero runtime dependencies; only
`aerointentbench.v2` uses numpy/Pillow (+rasterio for GeoTIFF), declared as the
optional `[v2]` extra and pinned by `tests/test_package_skeleton.py`. Nothing in the
V1 package imports V2.

## 4. Coordinate system

Local metric coordinates: origin at the raster's **top-left**, x right, y down, world
units in **meters**, image units in **pixels**; `meters_per_pixel` is the single scale.
Field names carry unit suffixes (`position_m`, `speed_mps`, `width_px`).

For the shipped GeoTIFFs (EPSG:3857) the scenario's `meters_per_pixel` is validated
against the raster's own transform (±5 %). Web Mercator carries the projection's
*nominal* metre — true ground metres differ by cos(latitude); V2 treats the nominal
metre as the local world metre and records the CRS/transform as provenance only. A
non-georeferenced image requires `meters_per_pixel` in the scenario.

## 5. The closed loop (canonical semantics)

Observations are scheduled every `observation_interval_s` (0.0, 1.0, 2.0 …). Per
processed observation: capture position ← `trajectory.position_at(capture_time)`;
render the crop; build the V1 `RuntimeState`; policy selects `config_id` (invalid
actions substitute the scenario's **explicit** `fallback_config_id` via the V1
`ActionValidator`); the executor runs on the RGB; `completion_time = capture_time +
mission_latency_s`; the UAV keeps moving (completion position is evaluated at
completion time); scheduled captures that passed while busy are recorded **skipped**
(never queued); the prediction is scored against the **capture-time** ground truth
(never a re-render at completion); battery = flight_power × elapsed + per-call energy;
terminate on path end / deadline / battery floor / unrecoverable executor failure.

Example: capture at 10.0 s with 2.4 s latency → completes 12.4 s; the 11.0 s and
12.0 s captures are skipped; the next processed capture is 13.0 s.

**Skipped-capture visibility (encountered-but-missed).** The executor never runs on a
skipped capture — no RGB is rendered for it and no prediction exists. But a target that
was visible *only* during skipped captures must not silently vanish from the mission
accounting, so the runner decides visibility at each skipped scheduled time
geometrically (the renderer's own inside-test on a sample grid over the footprint; no
raster read, no validity mask — diagnostic-only) and feeds it into the encountered set.
Consequences, pinned by `test_skipped_only_target_counts_as_encountered_but_missed`:
such a target counts as **encountered = true, detected = false,
missed-while-visible = true**, and the slow executor is penalized — `target_recall`'s
denominator is the scenario's total target count regardless, so recall was never
corruptible by skipping; the geometric check keeps the *diagnostics* honest too.

## 6. Synthetic objects and ground truth

Targets are **synthetic rescue-target markers, not realistic humans**: a
high-visibility orange body (optionally striped) that the lightweight executors can
find from RGB alone. Distractors wear similar-but-imperfect colours that fool the weak
gate but not the strong one. The source raster is never modified; objects are
composited per crop with rotation, boundary clipping, deterministic z-order, alpha
blending, and exact semantic (0/1/2 = background/target/distractor) and instance
(1-based scenario ordinal) masks. GT fields live on the `Observation` for the
evaluator only — the policy and the executor never see them.

## 7. Executors and honesty labels

`fast_weak` (downsample + coarse threshold, low configured latency, ragged, extra
false positives) versus `slow_strong` (full-res selective chroma + morphology +
small-component removal, high configured latency, clean masks). Every result labels
its quantities: `mission_latency_s` **simulated** (drives the clock),
`measured_wall_clock_s` **measured** (diagnostic only), `energy_j` **simulated**,
`communication_mb` **configured** (local, 0 in V2). Simulated values are never
presented as hardware measurements.

Extension point: a future heavy backend (MobileSAM / SAM2 / detector+SAM, FP16/INT8,
remote) implements `run(rgb) → ImageExecutionResult`, registers a new executor `kind`,
and lives in an optional module — the core never imports PyTorch.

## 8. Evaluation

The frozen V1 empirical vocabulary, unchanged: `target_recall` (unique targets found /
total, deduplicated by hidden object id — canonical), `detection_precision` (matched /
total predicted components), `false_positive_detections`,
`false_positives_per_processed_minute` and `false_positives_per_mission_minute`
(denominator-named), `target_f1 = null` (no persistent predicted-track identity).
Matching is the V1 discipline: IoU matrix, scenario threshold, greedy one-to-one.
V2 adds diagnostics: processed/skipped counts, targets encountered / detected /
missed-while-visible, path completion, termination reason.

## 9. Demo scenario and measured behaviour

`data/v2_scenarios/demo_img1_lawnmower.json`: img_1 (OpenAerialMap, 0.0295 m/px), a
conservative 80×30 m lawnmower ROI at (270–350, 225–255) m, 24×18 m footprint at
256×192 px, 3 synthetic targets + 3 distractors, two executors (0.4 s vs 2.5 s), one
`target_recall ≥ 0.6` contract, deterministic seed. Observed (synthetic, low-fidelity):

| policy | processed | skipped | recall | precision | FPs | battery | time |
|---|---|---|---|---|---|---|---|
| always_fast | 70 | 0 | 1.000 | 0.435 | 26 | 0.522 | 70.0 s |
| always_strong | 24 | **48** | 1.000 | **1.000** | 0 | 0.503 | 71.5 s |
| rule_based | 70 | 0 | 1.000 | 0.435 | 26 | 0.522 | 70.0 s |

The closed-loop trade-off is visible: the strong model skips two of every three
observations yet stays perfect on precision; the weak model sees everything and pays
in false positives. **These numbers validate the loop, not any perception model.**

## 10. Known limitations (deliberate)

No flight dynamics, wind, attitude, or path planning (the path is predefined; the
policy controls only the model-strategy). Orthographic camera, fixed altitude —
altitude does not affect the image. Latency/energy/communication are configured, not
measured. Targets are synthetic markers. The constant network is policy-visible
decoration. `rule_based` is a V1 policy reasoning over tiers — it is not tuned for V2.
Rasters are local-only (provenance in `data/v2_scenarios/aerial_sources.json`); large
files are gitignored.

## 10.1 V2.1 — real model-strategies (`torch_semantic_segmentation`)

V2.1 puts an **actual pretrained neural network** behind the executor seam. A
`torch_semantic_segmentation` executor config (validated strictly at scenario load)
names a torchvision model and is backed by real official weights:

- `local_light_real` → `lraspp_mobilenet_v3_large` (official DEFAULT weights)
- `local_strong_real` → `deeplabv3_resnet50` (official DEFAULT weights)

**Backend / executor split.** `TorchSegmentationBackend` builds the model once, loads
the official weights once, resolves the `person` class index **from the weight
metadata** (never hardcoded), places the model on the resolved device (`auto`: cuda →
mps → cpu, actual choice recorded), applies the official weight transform (preset
resize disabled where the API allows; identifier recorded), warms up outside mission
timing, and serves device-synchronised forward passes. Backends are **cached** per
(model, weights, device, dtype, input size) — weights never reload per frame.
`TorchSemanticSegmentationExecutor` resizes the RGB crop to the configured model input
(bilinear), thresholds the target-class probabilities (configurable), applies
config-declared postprocessing, and resizes the binary mask back to the observation
(nearest-neighbour). Its whole input surface is `run(rgb)` — GT and object metadata are
structurally unreachable.

**Latency semantics.** Two measured quantities are always recorded
(`model_forward_latency_s`, device-synchronised; `end_to_end_executor_latency_s`,
preprocess→postprocess) plus the mission latency. `latency_mode: "measured"` lets the
real end-to-end latency drive the mission clock (machine-dependent — stated in the
result); `"configured"` keeps the deterministic profile value with the measurement as a
diagnostic. Energy stays **simulated** (elapsed time is not a power meter);
communication stays configured (local).

**Dependencies.** `pip install -e ".[v2-real-models]"` (torch, torchvision), imported
lazily only when a torch executor is actually built; a missing install fails with that
exact command. The default test suite runs on an injected fake backend; the genuine
models are exercised by opt-in tests (`pytest -m real_models`) and by
`python -m aerointentbench.v2.cli check-real-models --scenario … [--load]`.

**Domain mismatch (deliberately reported).** The shipped scenario
(`demo_img1_real_models.json`) still uses **synthetic rescue markers**, which a
COCO/VOC-trained person model has never seen. Poor or empty masks there are a domain
mismatch, not an implementation failure and not a perception result; marker colours are
not tuned to exploit the pretrained models. Realistic aerial-person target assets (or
an aerial-person dataset) are required for the next *evaluation* milestone — V2.1 is an
*integration* milestone.

**Future strategies (documented, not built).** The same `run(rgb)` seam accommodates a
prompt-based pipeline — lightweight detector → person boxes → SAM/SAM2 → instance
masks — as a new executor kind; SAM alone is not an autonomous detector and is never
fed GT boxes in benchmark mode. Remote execution, INT8/FP16 profiles, and dynamic
networking remain future kinds/parameters on this seam.

## 10.2 V2.2 — image-based human target assets and controlled observability

V2.2 replaces "orange rectangle stands for a person" with an **image-asset target
layer**, so the V2.1 models can be shown something person-shaped — without ever bending
the evaluation toward them.

**Assets** (`aerointentbench/v2/assets.py`). A strict manifest (`asset_schema_version
"1.0"`) records, per asset: RGBA (alpha-derived mask) or RGB+mask-file, category,
**view_type** (`conventional` / `aerial` / `procedural` — never conflated, never
relabelled), full provenance (provider, license, redistribution permission), explicit
nominal physical dimensions, and a verified sha256. Missing files, empty masks,
dimension mismatches, unknown fields, and checksum lies all fail actionably. **No
licensed human asset ships with the repository** — `data/v2_assets/` holds the template
and policy; local, non-redistributable assets are gitignored.

**Rendering** (`ObjectLayer`, render_mode `image_asset` beside the preserved V2.0
`procedural_marker`). Physical metres → footprint-scale pixels decide the projected
size (a zero-pixel projection is recorded, never enlarged); RGB resizes bilinearly and
composites through the smooth alpha (anti-aliased edges, `opacity`,
RGB-only `brightness_factor`); the **binary GT mask travels the same spatial transform
with nearest-neighbour resampling** — ground truth is transformed geometry, never a
threshold of the composited pixels. Rotation, crop clipping, partial visibility,
overlap z-order, and per-object projection stats (`asset_projections` in observation
provenance) are all exact and tested. Executors and policies still see only the final
RGB.

**Controlled observability** (`aerointentbench/v2/observability.py`, CLI
`run-observability`). A deterministic single-target matrix — asset × projected size ×
rotation × background — rendered by the normal camera pipeline and scored per model:
positive pixels, target intersection / pixel IoU / pixel recall, false positives, and
detection under the existing evidence rule, plus measured latencies. Every report
carries `evaluation_purpose` and the view types present; `controlled_observability` /
`integration_diagnostic` results are never presentable as aerial-human perception.

**Measured so far (honest).** With a clearly-labelled *procedural* silhouette (blue
head-and-body shape, 0.3–1.2 m widths → 3×10 to 13×37 projected px on img_1), both
real models returned **empty person masks in all 18 conditions** (zero overlap, zero
false positives; LRASPP ~10 ms, DeepLabV3 ~175 ms forward on MPS). That is the
pipeline working and the domain gap being real: flat cartoon silhouettes at aerial
scales are invisible to generic COCO/VOC models. The conventional-view-cutout question
(outcome A) stays open until an owner supplies a **licensed** person asset — the
committed `observability_img1.json` runs it the moment `data/v2_assets/manifest.json`
exists. The V2.2 mission scenario is deliberately deferred until some asset is
observably detected (§16 gating of the milestone).

## 10.3 V2.2 결과 — synthetic generated human-target observability

Owner-supplied **OpenAI-generated synthetic humans** (5 originals under
`src/v2_img/human/`, classified by visual content: conventional front-standing, aerial
standing, aerial walking, aerial crouching, plus one rear-view standing registered as
AMBIGUOUS — possibly intended as lying — and excluded from the first experiment) were
deterministically cleaned (`aerointentbench/v2/asset_prep.py`: halo removal preserving
the 3 px anti-aliased band, bbox+8 px crop; originals untouched) and registered in the
strict manifest with `synthetic: true`, pose, processing provenance, and
redistribution **pending owner confirmation** (processed PNGs are local-only,
gitignored).

**Controlled observability** (footprint 6×4.5 m → 96/64/32 px ≈ 2.25/1.5/0.75 m):

- **Stage A (conventional sanity): PASSED.** Large/medium detected by both models
  (IoU up to 0.91); small (11×32 px) detected only by DeepLabV3.
- **Stage B (aerial): GATE PASSED — 21/24 detected.** DeepLabV3: **24/24**, IoU
  0.82–0.96, consistently fewer FPs. LRASPP: all large, but **0 at ≤32 px** for
  standing/walking (crouching survived at IoU 0.59). Background complexity (grass vs
  path) barely mattered. Latencies ~10 ms vs ~175 ms forward (mps/float32).
- **Interpretation: Outcomes C+D.** A genuine model-strategy trade-off now exists in
  the benchmark: the strong model buys small-target detection and precision with ~17×
  the latency. The earlier procedural-silhouette null result is superseded for
  *shape-realistic* synthetic targets.

**Gated mission** `demo_img1_generated_humans` ("synthetic generated aerial-human
mission diagnostic"): 3 aerial humans (64/43/28 px projected) + 1 distractor. Both
policies: recall 2/3 (both missed the rotated edge-of-lane walking target — recorded,
not tuned); precision **0.353 (light) vs 0.667 (strong)**, FP 11 vs 4.

All of this is **synthetic generated human-target observability — never real
aerial-human perception performance**.

## 10.4 V2.3 — replay bundle export and the static viewer

**Purpose.** Make one mission's configuration-selection story inspectable over time:
what the policy chose, what it cost, how the hard constraints evolved, and how the
mission ended. The viewer's information hierarchy is deliberate — (1) mission outcome,
(2) hard-constraint status, (3) policy/system behaviour, (4) perception diagnostics —
and perception must stay visually subordinate.

**Runtime snapshots** (`MissionRunner(record_runtime_snapshots=True)`, default off so
existing result files stay byte-identical): each `ObservationLog` gains a `runtime`
block — `at_capture` (exactly the policy-visible `RuntimeState`, never ground truth),
`after_completion` (battery_frac, cumulative_energy_j, cumulative_communication_mb,
remaining_deadline_s, path_progress), plus `config_switched` / `fallback_used` /
`action_reason`. The snapshot restates values the loop already computed — it is a time
series for replay consumers, never a second ledger.

**Replay bundle** (`aerointentbench/v2/replay_export.py`, `replay_schema_version
"1.0"`): `manifest.json` (identity, contract, executor configs, map geometry, honesty
labels), `events.json` (ordered observation timeline: snapshots, evaluator scores,
cumulative quality, constraint status, frame references), `frames/obs_NNNNNN_{rgb,pred,
gt}.png`, `overview.png`, `index.html`. Frames are re-rendered deterministically
through the mission's own renderer/executor (position is a pure function of mission
time), so nothing is stored during the run and nothing uses a second implementation;
the exporter *verifies* its cumulative tallies against the evaluator's and fails if
they diverge. Strict JSON (`allow_nan=False`), relative paths only; deterministic for
identical inputs except the documented `execution.measured_wall_clock_s` diagnostic.

**Constraint status** is a replay-time presentation layer over the existing semantics:
`SAFE` / `AT_RISK` / `VIOLATED` / `NOT_APPLICABLE` / `UNKNOWN`. At-risk margins are
simple and explicit (deadline <20% remaining; battery within 0.1 of the contract
floor; ≥80% of the communication budget spent); interim quality is progress, not a
verdict — the final event's statuses come from the evaluator's own constraint
booleans, and the viewer never runs a second success evaluator. **Privacy is
`NOT_APPLICABLE`**: V2 declares no remote configuration or executor, so no executable
path can violate it. TODO (blocking for remote work): align V1/V2 privacy semantics —
including a privacy branch in the V2 mission-success conjunction — before any remote
executor or remote configuration is introduced.

**Viewer** (`replay_viewer.py`): one self-contained HTML page, plain HTML/CSS/JS, no
framework, no build step; the replay JSON is embedded in the page and frames load by
relative path, so it works from `file://` (or `python -m http.server --directory
<bundle>`). Panels: a **map-dominant layout** — the mission map (top, ~2/3 width) over
the current perception view (RGB + prediction overlay; ground truth only behind an
explicit **Debug GT** toggle), with the mission dashboard below in the hierarchy above.
Timeline: prev/next, play/pause, slider, speed control (0.5–8×), and a config-selection
strip over mission time.

**Map background & smooth playback** (additive `manifest.map` fields, still
`replay_schema_version "1.0"`; added before any release and the viewer falls back to
schematic drawing / discrete stepping when absent):

- `map.background` — `{file, extent_m: [x0,y0,x1,y1], width_px, height_px}`: a
  pre-rendered crop of the mission's own world raster (through the same
  `read_window_m` the camera uses), so the trajectory registers exactly on the real
  mission environment. The extent derives from the trajectory padded by one camera
  footprint — never from ground-truth object placement. No tiles, no online map
  service; the bundle stays offline-capable.
- `map.drone_speed_mps` + `map.trajectory_duration_s` — the viewer's interpolation
  basis. Playback advances a **mission-time clock** (`requestAnimationFrame`; 1× =
  real mission seconds) and interpolates the UAV position with the simulator's own
  rule (constant speed along the waypoint polyline, clamped), while observation
  panels, detections, and skip markers switch exactly at their event times. A test
  pins that this interpolation reproduces every logged capture/completion position.
  Smooth motion is presentation only — mission semantics live in the events.

Usage:

```bash
python -m aerointentbench.v2.replay_export \
  --scenario data/v2_scenarios/demo_img1_generated_humans.json \
  --policy rule_based --output results/replay_demo
# equivalently: python -m aerointentbench.v2.cli export-replay ...
```

## 10.5 Hard trade-off scenario — where configuration selection decides the mission

`data/v2_scenarios/demo_img1_hard_tradeoff.json` (`V2_IMG1_HARD_TRADEOFF`) is the first
V2 scenario in which **no static configuration can succeed and an adaptive policy
can** — the benchmark's core claim, made concrete with real pretrained models.

**Design** (all invariants pinned by `test_shipped_hard_tradeoff_scenario_pins_its_design`):

- Camera footprint 6×4.5 m at 256×192 px — the exact operating point validated by the
  V2.2 Stage B observability matrix (42.7 px/m).
- **Four small early targets** (0.75 m ground extent → 32 px, standing/walking): Stage B
  showed LRASPP detects nothing standing/walking at ≤32 px while DeepLabV3 does. They
  sit on the first two lanes with visibility windows covering even capture times.
- **Four large late targets** (64–96 px): detectable by the light model, but their
  ~1.5 s visibility windows (6 m footprint / 4 m/s) are centred on **odd** capture
  times on the last lane. The strong executor's configured 1.4 s latency gives it a
  2-capture cadence, so a static strong policy skips every one of them
  (encountered-but-missed, the pinned V2 skip semantics).
- **Stress-configured energy** (`energy_j_per_call`: strong 600 J — simulated,
  modelling a power-hungry onboard accelerator): an always-strong policy also breaches
  the 0.22 battery floor. Latency and energy are configured/simulated
  (`latency_mode: "configured"`), so mission outcomes are deterministic and
  machine-independent; wall-clock stays a diagnostic.

**Measured outcomes** (local run, `results/v2_hard/`, torchvision official weights,
mps device — outcomes deterministic given the configured clock):

| policy | recall | battery | mission | why |
|---|---|---|---|---|
| `always_light_real` | 0.500 (4/8) | 0.748 | **FAIL** | quality: misses all 32 px targets (plus the −30° walker); 56 false positives |
| `always_strong_real` | 0.500 (4/8) | 0.180 | **FAIL** | quality **and** battery: skips all four odd-window targets, drains the pack |
| `rule_based` | **0.750 (6/8)** | 0.227 | **SUCCESS** | strong early (catches the small targets), one battery-pressure switch to light at ~t=42 s (catches E3/E4) |

The adaptive win is emergent, not scripted: `rule_based` sees only the contract and
the policy-visible state; its single switch comes from its battery-pressure rule
(battery ≤ floor + 0.05). Honest footnotes: the light model *did* catch one 32 px
standing target and missed one rotated 64 px walker — model reality, recorded as-is,
and the trade-off holds regardless. The rule-based final battery clears the floor by
only ~0.7 pt; that tightness is deliberate (the scenario is meant to punish a late
switch) and stable because the mission clock is configured. Replay bundles for all
three policies live under `results/v2_hard/replay_<policy>/` (local-only, composited
human frames are never committed). As everywhere in V2.2: **synthetic generated
human-target observability — never real aerial-human perception performance.**

## 10.6 V2.4 — multi-seed hard-scenario evaluation

**Question.** Is "adaptive beats static" a reproducible pattern across controlled
variations, or one hand-designed episode? (`scenario_family.py`, `multi_seed_eval.py`)

**Method.** A deterministic scenario family derives variants of
`demo_img1_hard_tradeoff` as a pure function of `(FAMILY_VERSION, base id, seed)` —
sampled: small-target slots/jitter/size/pose/rotation/lateral offset on the early
lanes, late-target jitter/size/pose/rotation/lateral offset on the {41,43,45,47} s
post-switch slots, battery capacity in [6.60, 7.00] Wh. NOT sampled: executors,
contract, trajectory, camera, world. Every sampled value is recorded in
`provenance.scenario_family`; a sampling change bumps `FAMILY_VERSION`. All policies
run the **same** scenario instances (paired by construction); per-run records restate
the evaluator's outputs; success rates use the V1 Wilson interval.

**Family 1.0 lesson (kept as sensitivity evidence).** The first sampling scheme let
late targets appear from ~37 s — before the adaptive policy's battery-pressure switch
(~40-43 s) — and its ±0.2 s jitter let wide targets clip the edge of a
strong-cadence frame. A 5-seed smoke run collapsed to 4/5 all-fail: **the adaptive
advantage is sensitive to the alignment between late-target onset and the switch
time.** Family 1.1 realigns the late window with the documented hard-scenario
structure (≥ 41 s, ±0.1 s jitter); the 1.0 finding is part of the result, not a
discarded draft.

**Results — family 1.1, seeds 0-29, 90 paired runs** (local, torchvision official
weights, configured latency → outcomes deterministic; `results/v2_hard_multiseed/`):

| policy | success | rate | Wilson 95% CI | dominant failures |
|---|---|---|---|---|
| `always_light_real` | 0/30 | 0.000 | [0.000, 0.114] | quality 100% |
| `always_strong_real` | 0/30 | 0.000 | [0.000, 0.114] | battery 100%, quality 97% |
| `rule_based` | **19/30** | **0.633** | **[0.455, 0.781]** | quality 30%, battery 17% |

Mechanism (mean per-class detection rate): small targets — light 0.05, strong 1.00,
rule 1.00; late targets — light 0.77, strong 0.09, rule 0.45. The complementarity is
exactly the designed trade-off, and rule's 0.45 late rate (vs light's 0.77) is the
honest price of switching late. Paired outcomes: **19 seeds only-adaptive-succeeds,
11 all-fail, 0 seeds where any static succeeds, 0 counterexamples** where adaptive
loses to a static. Rule's 11 failures: 9 quality (missed late targets before/at the
switch), 5 battery (margin −0.006 min), 3 both.

**Fragility flags (raised by explicit rules, not narrative):** (1) 6/19 adaptive
successes clear the battery floor by <0.01 — the family is knife-edged on battery by
design, but a small energy perturbation flips those missions; (2) every adaptive
success uses exactly **one** strong→light switch — the result demonstrates the value
of a single battery-pressure adaptation, not of rich adaptive behaviour.

**Supported claims (this family only):** on 30 paired seeds, the adaptive policy
succeeds significantly more often than either static policy (non-overlapping Wilson
CIs); no static policy ever satisfied the contract; no counterexample seeds exist.
**Not supported / not evaluated:** generalisation beyond this family (other worlds,
contracts, trajectories, network dynamics); anything about real aerial-human
perception; robustness of policies richer than one switch; publication-grade sample
size (30 seeds is development-grade; the command scales to 100+ at ~13 s/run).

Reproduce:

```bash
python -m aerointentbench.v2.multi_seed_eval --seeds 0:29 \
  --output results/v2_hard_multiseed          # ~20 min on Apple Silicon (mps)
```

## 11. V2 freeze

V2 closes with V2.4. Freeze checklist (all verified at freeze preparation):

- [x] full default test suite passes without torch/local assets (opt-in markers
      `real_models` / `real_assets` stay deselected)
- [x] `ruff check` and `ruff format --check` clean
- [x] deterministic scenario generation (same seed → identical bytes; pinned by test)
- [x] repeatable experiment command with resume (`multi_seed_eval`, chunk-safe)
- [x] strict output serialisation everywhere (`allow_nan=False`; wall-clock excluded
      from experiment records)
- [x] no GT leakage into policy-visible state (pinned by tests at every layer)
- [x] replay-viewer bundles compatible (family variants export unchanged)
- [x] documentation: scope, methodology, statistics, limitations, licensing,
      privacy status (NOT_APPLICABLE + pre-remote TODO), supported vs unsupported
      claims (this section and §10.1-10.6)
- [x] restricted assets untracked (human PNGs, rasters, generated frames all
      gitignored; redistribution PENDING OWNER CONFIRMATION)
- [x] clean working tree at the freeze commit; no push, no merge without owner
      authorisation

Claim-status vocabulary for anything citing V2: **demonstrated in one scenario**
(§10.5 reference episode) / **supported across evaluated seeds** (§10.6, 30 seeds) /
**not yet evaluated** (other families, network dynamics, richer policies) / **out of
scope for V2** (remote execution, learned policies, physics, real-human data).

## 12. Next steps (V3 and beyond — out of V2 scope)

**Roadmap (owner decision, 2026-07-29): V3 = benchmark completion; the 3D simulator
moves to V4.** V3 closes the gaps V2.4's own fragility analysis exposed, in order:

1. **Activate the dormant contract axes** — a remote executor kind, dynamic network
   (reuse V1's `TraceBasedNetworkModel`), and V1/V2 privacy alignment (the documented
   pre-remote TODO, including a privacy branch in V2 mission success). This makes
   communication and privacy real constraints and wakes the rule-based policy's
   dormant reachability/affordability/latency-budget rules.
2. **Statistical hardening** — 100-seed main evaluation, a second world (`img_2`),
   wider family dimensions; retire the "every success is one switch" flag.
3. **Policy skyline** — an offline-optimal (GT-aware upper bound, clearly labelled)
   and a simple budget-planning policy, so `rule_based` has context above it.
4. **Real-data grounding** — one real aerial-person dataset through the completed V1
   empirical bundle pipeline (external dependency: dataset licensing).

Toward real models: implement a heavy `ImageExecutor` kind in an optional module
(e.g. MobileSAM), measure real wall-clock as *its own labelled quantity*, and feed
per-config measurements through the existing V1 empirical bundle builder for
reproducible replay. Toward **V4**: replace `WorldSource`/`Trajectory`/`CameraRenderer`
with a 3D simulator adapter (AirSim / Isaac / Gazebo) behind the same interfaces; the
runner, policy interface, and evaluation are designed to survive that swap.

## 10.7 Periodic ground-station telemetry (2026-08-06)

A real search UAV streams mission-progress reports (position, battery, status,
detection summary) to its ground station even when perception runs fully local; the
closed loop previously generated communication only when a remote configuration was
selected. The optional `simulation.telemetry` block adds that background uplink:

```json
"telemetry": {"interval_s": 1.0, "report_mb": 0.001,
              "energy_j_per_mb": 60.0, "max_loss_frac": 0.5}
```

Semantics (pinned by `tests/test_v2_telemetry.py`):

- Reports are scheduled on mission time (`interval_s`, first at one interval) and
  each attempt is evaluated against the **capture-time network sample at its own
  scheduled time**: sent when the uplink is up and packet loss is within
  `max_loss_frac`, otherwise **lost — nothing transferred, nothing charged**, the
  loss only counted.
- Sent report MB join the mission communication ledger and therefore **share the
  contract's `communication_budget_mb`** — telemetry can genuinely crowd out
  offloading (or, sized aggressively, fail the constraint on its own). Transmit
  energy (`report_mb x energy_j_per_mb`) is charged to the battery but ledgered
  separately from remote-inference transfer energy.
- Reports never block or delay the perception loop (no latency charge), and are
  asynchronous fire-and-forget: no retransmission, no queueing across an outage.
- **A scenario without the field has no telemetry and byte-identical results** —
  every existing scenario and pinned outcome is untouched.

The result record gains `telemetry_reports_sent / _lost`, `telemetry_mb`, and
`telemetry_energy_j` (additive fields, zero when disabled). The hardware-grounded
zoo profiles (`docs/v3x_extensions.md` §4) enable 1 Hz reports sized on the measured
353 B protocol envelope (~1 KB with status headroom); their transmit energy reuses
the configured 60 J/MB radio stress value, which remains unmeasurable on the devkit
rails (B4). This is telemetry only — split computing / partial-feature offload
remains out of scope and undesigned.

## 10.8 Split inference (2026-08-06)

The third deployment option beside full-onboard and full-server: the
``simulated_split`` executor kind runs the model's first stages (the *head*)
onboard, ships the intermediate activations across the simulated link, and runs
the remainder (the *tail*) server-side. Semantics (pinned by
`tests/test_v2_split.py`):

- **Graph-cut payload**: the wire payload is every tensor crossing the cut — the
  running activation plus any backbone tap already produced (LRASPP's low/high
  taps are priced, never under-counted) — sized from the actual arrays at runtime,
  never configured. Feature reduction (``feature_dtype``: float32/float16/uint8
  per-tensor affine) is applied to the tensors the tail actually consumes, so its
  accuracy cost is real. Learned bottleneck compression is out of scope (training).
- **One transport, one formula**: the split executor subclasses the remote
  executor through a payload-preparation hook; stage timing, failure precedence,
  upload-charged-on-failure, and same-frame fallback all remain the single
  documented implementation. Head latency (``head_latency_s``) and head energy
  (the config's ``energy_j_per_call``) are charged on every attempt — a failed
  upload wastes real onboard work.
- **Privacy is derived from the kind, never author-declared**: the catalog maps
  ``simulated_split`` to REMOTE placement with ``transmitted_payload="features"``
  — legal under ``features_only`` (where raw-RGB offload is a violation),
  forbidden under ``local_only``. This is the structural reason split exists:
  measured JPEG sizes made full-frame offload bandwidth-cheap, so split's niche
  is privacy plus onboard-compute relief, not bytes.
- **Real cuts are exact**: ``split_models.TorchSplitPartition`` partitions the
  cached torchvision model at a legal backbone-child boundary (illegal cuts fail
  loudly, listing the legal set); head+tail execute exactly the full model's
  operations, so a float32 cut reproduces the full model's mask bit-for-bit
  (pinned by the opt-in ``real_models`` test). The heuristic partition is a CI
  fixture only. Onboard-full reduced precision needs no new machinery: the torch
  kind's existing ``dtype: float16`` is the quantized tier (int8/TensorRT stays
  out of scope).

## 10.9 Detection-evidence reports (2026-08-07)

"The user must see what the drone found, regardless of where inference ran." The
telemetry block's optional ``evidence_mb_per_detection`` (default 0.0 = off) adds
that traffic: when an observation completes with N predicted components, one
evidence transmission of ``N x evidence_mb_per_detection`` MB is attempted at
completion time against the capture-time network sample. Semantics (pinned by the
evidence tests in `tests/test_v2_telemetry.py`):

- The count is the drone's OWN ``predicted_components`` (the evaluator's
  prediction-side tally — no ground truth involved): **false positives spend real
  communication**, so an inaccurate model taxes the link as well as the metric
  (in the deployment matrix, light-fp32's FP burden sends 6.6x the evidence MB of
  the accurate strong-fp16).
- Sent evidence shares the contract's communication budget and charges transmit
  energy to the battery; evidence attempted during an outage is **lost and free**,
  and ``evidence_reports_lost`` records the **user-visibility gap** — detections
  the mission made but the user never saw. Fire-and-forget: no queueing or
  retransmission across outages (a deliberate v1 semantic; a later contract
  constraint could bind on the gap).
- Absent field / zero -> byte-identical behaviour; all result fields are additive.

The deployment-matrix scenario grounds the size on measurement: 0.003 MB per
detection = the measured RLE mask wire size (B2). Whether user-visibility becomes
a sixth contract constraint is an open owner decision, recorded here.

## 10.10 Continuous observation stream (2026-08-07)

"The drone should just keep sending what it sees." It can — as the third downlink
layer, ``stream_mb_per_observation`` in the telemetry block: one frame of that size
is attempted at **every capture-cadence tick on the mission clock** (including
slots the busy executor skipped — the camera still saw them). Sent frames share
the communication budget and charge transmit energy; frames attempted during an
outage are lost and free, and ``stream_frames_lost`` is the **operator's blind
time**. Sized honestly, a 512x384 JPEG frame is the measured 0.028 MB (B1).

Two consequences the stream makes explicit rather than hiding:

1. **Privacy**: the stream is imagery leaving the vehicle — however downscaled,
   compression is not de-identification — so a scenario may enable it only under
   ``remote_allowed`` (load-time error otherwise). Under ``features_only`` /
   ``local_only`` the user's view is status + evidence, which is what those
   privacy levels *mean*.
2. **It does not dissolve the placement question.** A situational stream gives the
   user awareness; it does not produce masks. Full-quality streaming plus
   server-side perception is exactly what the ``simulated_remote`` config already
   models, with its costs. And on the shipped network traces the stream dies in
   the disconnected regimes — the blind-time counter records precisely when
   onboard autonomy is the only thing still working.

The three downlink layers compose: **stream** (what the drone sees, ambient,
remote_allowed only) / **status telemetry** (that the drone is alive, tiny,
always) / **evidence** (what the drone found, event-driven, prediction-derived).

## 10.11 Model catalog: predefined split as a fixed action (2026-08-11)

The catalog a policy selects from is `model_family x execution_mode`: the same
architecture may appear as full onboard (torch kind, fp32), reduced-precision
onboard (the torch kind's `dtype: float16`; int8 needs TensorRT and stays out of
scope), full raw-RGB offload (`simulated_remote`), or a split deployment. The
policy's action stays exactly one `config_id` — execution mode is a property of
the catalog entry, never something the policy composes at runtime.

The `pretrained_split` executor kind (`aerointentbench/v2/presplit.py`) adds
splits **published by prior research**: an onboard head (encoder) and a server
tail trained together by a split-computing paper and distributed as pre-trained
checkpoints. Semantics (pinned by `tests/test_v2_presplit.py`):

- **The split point is a citation, not a search result.** This benchmark's
  research question is configuration *selection*, not split-point discovery, so
  layer-wise profiling, split search/optimization, and split-aware retraining are
  all out of scope. Every `pretrained_split` config carries a mandatory
  `SplitSpec` — `split_source` (`paper` / `official_repository`; there is
  deliberately no value meaning "found by this benchmark"), a citable
  `split_source_reference`, and the `split_location` in the source's own terms —
  and a config without provenance fails at scenario load.
- **One transport.** The executor subclasses the remote executor through the same
  `PreparedPayload` hook as the graph-cut split: stage timing, failure
  precedence, upload-charged-on-failure, and same-frame fallback remain the
  single documented implementation. Head latency (configured, pending board
  measurement) and head energy (`energy_j_per_call`) are charged on every
  attempt.
- **Payload honesty.** The wire size is the byte length the head actually encodes
  for this frame — for the sc2 backend, the entropy-coded bitstream — never a
  configured constant.
- **Privacy is derived from the kind**: REMOTE placement with
  `transmitted_payload="features"` — legal under `features_only`, forbidden under
  `local_only`, via the frozen V1 privacy logic.
- The genuine backend (`sc2_entropic_student`) consumes SC2-benchmark checkpoints
  (Entropic Student DeepLabV3-R50, VOC; MIT license; local-only cache like the
  model-zoo weights) behind the `[v2-presplit]` extra, lazily imported; the
  default suite uses a labelled fixture and downloads nothing. Training or
  fine-tuning bottlenecks ourselves stays banned — published checkpoints are
  consumed exactly like torchvision weights.

Demo: `data/v2_scenarios/demo_img1_model_catalog.json` — the deployment-matrix
mission re-expressed as a catalog: deeplabv3_resnet50 x {onboard fp32, onboard
fp16, raw remote (privacy-blocked under the features_only contract), predefined
split beta 0.64 / 5.12}, lraspp x {onboard fp32, fp16} (no published split for
LRASPP was found; inventing one is forbidden, so the family honestly lacks the
tier). Measured on this world's frames at 512x384 input: entropy-coded payloads
12.6 KB (beta 0.64) / 1.1 KB (beta 5.12) — versus 3.5 MB for the naive uint8
graph cut of the same architecture (§10.8), the concrete argument for adopting
published splits instead of cutting graphs ourselves. Verified with genuine
checkpoints: both presplit missions complete with battery 0.71 (the tiny head
does what split computing promises) but fail quality (recall 0.375 / 0.25 —
VOC-trained checkpoints on synthetic aerial markers are the documented V2.1
domain mismatch; the non-zero recall is incidental marker-to-person firing,
never perception evidence).

### 10.11.1 Second method family and second model family (2026-08-11, same day)

Two catalog expansions, both still literature-fixed splits:

- **`sc2_ghnd_bq`** — the SC2 release's second published method family: the GHND
  bottleneck (Head Network Distillation, Matsubara et al., IEEE Access 2020;
  bottleneck architecture from "Neural Compression and Filtering...", 2021) with
  8-bit bottleneck quantization and NO entropy coding. The v0.0.3 checkpoints
  predate today's sc2bench module list, so the backend reconstructs the
  release-era architecture verbatim from the v0.0.3 source (transcription, not
  design — the checkpoint defines the computation). Measured on the catalog
  world: 39.6 KB/frame (vs 12.6 KB entropy-coded — the honest cost of skipping
  the entropy coder), and recall 0.625, the best of the split tiers.
- **`fcm_maskrcnn_fpn` + `torch_instance_segmentation`** — a second MODEL family
  (instance segmentation): torchvision Mask R-CNN R50-FPN official COCO weights,
  deployable full-onboard (new local kind, fake-injectable for CI, person index
  from weight metadata) or split at the MPEG FCM standard FPN test point
  (backbone+FPN onboard, P-layer features across, RPN+ROI heads remote). The
  standard's learned feature codec (FCTM/VVC) is out of scope, so crossing
  tensors reuse the graph-cut split's per-tensor affine quantization — measured
  4.19 MB/frame, and the genuine mission fails the communication budget: the
  payload gap between a standard split point without a learned codec and a
  paper's supervised compression is a benchmark finding. A float32 FCM split
  reproduces the onboard model's mask exactly (pinned by the opt-in test).

Not shipped, with reasons recorded: SAM/SAM2 split (promptable segmentation
needs a prompt policy the mission loop does not define — a design question, not
a wrapper); Ladon (research repo is not packaged; vendoring research code into
the benchmark violates dependency hygiene — revisit if it is released as a
package); int8 onboard (needs TensorRT, banned; fp16 stays the reduced-precision
tier).
