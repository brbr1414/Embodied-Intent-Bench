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

## 11. Next steps

Toward real models: implement a heavy `ImageExecutor` kind in an optional module
(e.g. MobileSAM), measure real wall-clock as *its own labelled quantity*, and feed
per-config measurements through the existing V1 empirical bundle builder for
reproducible replay. Toward V3: replace `WorldSource`/`Trajectory`/`CameraRenderer`
with a 3D simulator adapter (AirSim / Isaac / Gazebo) behind the same interfaces; the
runner, policy interface, and evaluation are designed to survive that swap.
