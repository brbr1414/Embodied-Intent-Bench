# CLAUDE.md — AeroIntentBench

Guidance for Claude Code sessions in this repository. Read this before changing anything.

## What this project is

**AeroIntentBench** is a benchmark for evaluating **intent-conditioned, resource-aware
inference configuration selection policies** in UAV missions.

A policy observes a structured mission contract and a policy-visible runtime state, then
selects one inference configuration from a fixed pool. The runner executes or simulates
that configuration, updates time, battery, network usage, and mission evidence, and
computes mission-level metrics.

The core question: *Can a policy select appropriate perception inference configurations
under changing battery and network conditions while satisfying a mission contract?*

The benchmark does **not** evaluate path planning, navigation, flight control, or
open-ended language understanding.

(The repository is named `Embodied-Intent-Bench`; the Python package and the benchmark
are `aerointentbench` / AeroIntentBench.)

## Current status

**V1 complete and runnable.** All eight branches in `docs/branching.md` are done. The
benchmark runs three 900-step episodes on CPU with zero runtime dependencies and writes a
metrics file; re-running produces a byte-identical result.

Before changing anything, read `docs/v1_spec.md` §16 (known limitations) — several are
deliberate and already decided, and re-litigating them wastes a session. The first one worth
fixing is `initial_altitude_m`, which the episode schema carries and nothing reads.

**`tests/reference/baseline_results.json` pins what every baseline scores.** If a change
moves those numbers, that is the benchmark moving. Regenerate with
`python -m tests.test_reference_suite --update` and review the diff deliberately; do not
regenerate to make a test pass.

**Never quote a success rate from three episodes.** Over the shipped suite a rate can only
be 0, 1/3, 2/3 or 1, and 2/3 carries a 95 % interval of [20.8 %, 93.9 %]. An earlier
revision of the spec claimed adaptation beat every static baseline on that basis; pooled
over seeds the difference was not significant, and the quality threshold had to be raised
to make it so. Use `--repeats` and report `mission_success_ci_95`. See `docs/v1_spec.md`
§12.

## Documents

| Document | Read it for |
|---|---|
| `docs/v1_spec.md` | Normative V1 scope, schemas, semantics, metrics, assumptions |
| `docs/architecture.md` | Module boundaries, interfaces, dependency rules, anti-patterns |
| `docs/branching.md` | Branch workflow and the V1 branch sequence |
| `data/README.md` | Fixture layout and the synthetic-data rule |

## V1 scope

In scope: UAV on a predefined path; fixed nominal altitude and velocity; one perception
task family (segmentation); one mission type (human search); one-second decision
interval; policy action = one predefined configuration ID; dynamic battery, bandwidth,
RTT, packet loss, remaining deadline, accumulated communication, path progress, and
accumulated evidence; profile-driven deterministic simulation; local **and** remote
configurations representable.

## Do not add these yet

Adding any of the following without an explicit request is a defect, not initiative:

- Real segmentation model execution (interface stub only), or any heavy ML dependency.
- GPU requirements. **The benchmark must run on CPU with zero runtime dependencies.**
- Gazebo, PX4, ROS 2, real UAV control, physical flight dynamics, navigation policy.
- Split inference, training pipelines, RL implementations.
- Speculative frameworks: dynamic plugin discovery, entry-point systems, deep inheritance
  trees, abstractions with no current implementation.

Future extensions (other tasks, missions, strategies, platforms, policies, simulators) may
**influence interface design** — see `docs/architecture.md` §9 — but must not be built.

## Architectural rules

**The benchmark core must contain no logic specific to human-search segmentation.**

- `EpisodeRunner` depends on protocols (`Policy`, `Executor`, `TaskEvaluator`,
  `EvidenceTracker`, `BatteryModel`, `NetworkModel`, `TerminationCondition`), never on
  concrete task code. It must not branch on `task_id` or contain mask/IoU logic.
- `task_id` resolves a registered `TaskDefinition`; executors and policies resolve through
  registries. Registries are added by the branch that has a second implementation to
  register, not before.
- **`config_id` is an identifier only.** Never parse it to infer behaviour. Behaviour comes
  from typed strategy metadata.
- Metrics consume a standardised `TaskEvaluationResult`; mission-success logic must never
  name a task-specific metric such as `target_f1`.
- Composition happens in `run_benchmark.py`; components are injected via constructors.
- Every specification and result file carries `schema_version`; V1 supports `"1.0"` only
  and fails loudly on anything else. Validation is strict — reject unknown fields.

**Policy-visible / simulator-internal / hidden ground truth are three separate tiers.**
A policy receives a frozen `RuntimeState` — never the `Episode`, the runner, the future
network trace, or anything ground-truth-derived. A ground-truth quantity reaching a policy
invalidates the benchmark.

## Coding conventions

- Python 3.11+, fully typed, `from __future__ import annotations`.
- **Zero runtime dependencies.** stdlib `dataclasses` (frozen where practical), not
  Pydantic. `pytest` is the only dev dependency.
- `pathlib` over `os.path`; stdlib `logging` (no bare `print` outside the CLI).
- JSON-serialisable typed records; deterministic seeds; pure functions where practical.
- `ruff` is a dev dependency and the codebase is clean under `ruff check` and
  `ruff format --check` (line length 100). Keep it that way.
- Docstrings state the module's **responsibility and boundaries**, not just its contents.

## Determinism

Given the same fixtures, seed, and policy, an episode must produce byte-identical metrics.
No wall-clock time, no unseeded randomness, no set/dict iteration order dependence, no
filesystem ordering dependence. Tests must not require network access, GPU hardware, or
external datasets.

## Synthetic data

All profile numbers, traces, predictions, and ground truth under `data/` are **synthetic
placeholders, not empirical measurements.** Keep that label visible in fixture names and
documentation. Never present them as hardware results.

## Branch workflow

- **Never commit to `main`.** One integration branch per milestone: **`develop/v3` is
  the current integration branch**; `develop/v1` (V1) and `develop/v2` (V2.0–V2.4) are
  frozen at their milestone freeze points and must not advance again.
- Every feature branch is created from the **latest current integration branch**. A
  branch name containing a slash is just a name, not a namespace — there is no branch
  hierarchy.
- Implement only the current branch's responsibility; run the tests; commit logical,
  reviewable changes using conventional commit messages.
- **Do not merge into an integration branch or `main`, and do not push, without
  explicit authorisation from the owner.**

## V2 (visual closed loop)

`aerointentbench/v2/` is the optional visual simulator (`docs/v2_design.md`): a GeoTIFF
world read window-by-window, a predefined trajectory, position-dependent crops, synthetic
target markers with exact GT, two RGB-dependent lightweight executors, and a closed loop
where configured latency moves the UAV and skips observations. Rules that must hold:

- **V1 stays frozen and zero-dep.** Nothing outside `aerointentbench/v2/` may import
  numpy/Pillow/rasterio (pinned by `test_package_skeleton.py`); the V1 package never
  imports v2 eagerly. V2 deps live in the `[v2]` extra only. PyTorch stays banned
  everywhere; a future heavy executor is an optional module behind the `ImageExecutor`
  seam.
- **GT boundary**: `Observation` ground truth is evaluator-only. Policies see the V1
  `RuntimeState`; executors see `rgb` and nothing else.
- **Closed-loop semantics are canonical**: capture-time GT scoring, no queued stale
  observations, position as a pure function of mission time.
- **Honesty**: V2 latency/energy/communication are simulated/configured, targets are
  synthetic markers — never present V2 numbers as real UAV perception performance.
- Large rasters (`src/v2_img/*.tif`) are gitignored and local-only; provenance lives in
  `data/v2_scenarios/aerial_sources.json`. Do not rename/delete the local files.

**V2.1 real models** (`aerointentbench/v2/real_models.py`, `docs/v2_design.md` §10.1):
`torch_semantic_segmentation` executors run actual pretrained torchvision models
(LRASPP-MobileNetV3 light / DeepLabV3-ResNet50 strong, official DEFAULT weights) behind
the same `run(rgb)` seam. Rules: torch/torchvision live ONLY in the `[v2-real-models]`
extra, imported lazily inside the backend (the default suite uses an injected fake and
never downloads weights; genuine models are `pytest -m real_models` / `check-real-models`);
the person class index comes from weight metadata, never hardcoded; backends are cached
per strategy (no per-frame reloads); `latency_mode` measured vs configured is explicit and
never silently mixed; energy stays simulated. **Synthetic markers ≠ people**: COCO/VOC
models scoring recall 0 on the V2 scenario is documented domain mismatch — do not tune
marker colours to exploit pretrained models, and never present V2.1 numbers as real
aerial-human perception. `.venv` may be a symlink to `~/.venvs/aerointentbench` (large
venvs inside the OneDrive-synced tree cause file-provider stalls).

**V2.2 image-asset targets** (`aerointentbench/v2/assets.py`, `observability.py`,
`docs/v2_design.md` §10.2): image targets need a strict manifest with truthful
provenance — provider, license, `redistribution_allowed`, verified sha256, explicit
physical dimensions, and a `view_type` that is never relabelled (`conventional` /
`aerial` / `procedural`). Hard rules: **no scraped or unknown-license person imagery,
ever**; no licensed human asset is committed (local ones are gitignored); GT always
comes from the transformed asset mask, never from thresholding composited RGB; a
zero-pixel projection is recorded, never enlarged; procedural silhouettes are test
fixtures, never person-performance evidence; `evaluation_purpose`
(`controlled_observability` / `integration_diagnostic`) results must never be presented
as aerial-human perception. The V2.0 `procedural_marker` mode stays untouched.

**V3 P1 remote inference + dynamic network + privacy** (`aerointentbench/v2/remote.py`,
`network.py`, `docs/v3_design.md`): the `simulated_remote` executor kind runs behind
deployment-ready boundaries (`InferenceTransport` / `RemoteInferenceBackend` /
`run_with_context`) — a future Jetson client + real server replaces the simulated pair
without touching runner/policy/evaluator. Rules: remote latency is DERIVED (Model A:
capture-time network snapshot; formula and per-stage breakdown in docs — never one
opaque constant); uploads are charged even on failure (partial transfers
proportionally); communication energy = configured J/MB + activation, charged on every
attempt, distinct from compute/flight energy; a failed remote attempt is
`success=False` + status, never a silent empty prediction; executor-level fallback runs
on the SAME captured frame with both attempts on the mission clock; the scenario-level
safe fallback must be local. `network_trace` = named piecewise-constant regimes; the
policy sees only the current sample as the frozen V1 `NetworkObservation` (no regime
names, no futures). **V2 mission success is now the 5-constraint AND incl. privacy**
(V1 `privacy_permits` verbatim: remote raw-RGB configs are forbidden under `local_only`
AND `features_only`; blocked selections are counted violations). Local-only scenarios
keep their outcomes (pinned). Protocol models are wire-representable
(`protocol_version "1.0"`). No real server/RPC/Jetson in this milestone; nothing here
is a hardware claim.

**V3 P4 real-data pilot** (`experiments/real_segmentation_pilot/uavid.py`,
`torchvision_models.py`, `uavid_pilot.py`, `docs/v3_design.md` §P4): UAVid (real
oblique UAV imagery, CC BY-NC-SA, local-only — NEVER committed) ran through the
UNCHANGED V1 empirical chain (run_inference → build_pilot → build/validate bundle →
`run_benchmark --executor replay`). 12 keyframes / 164 derived person instances;
measured recall LRASPP 0.030, DeepLabV3 0.122 — an expected DOMAIN-MISMATCH
diagnostic, never attainable-perception evidence. Honesty rules this pilot added:
person instances are DERIVED (connected components of the semantic Humans class;
touching people merge; no temporal identity → per-instance recall), evaluation uses
1280x720 native windows because the frozen dense-mask wire format makes full-4K
crowded frames multi-GB (measured), out-of-domain models need a documented
physical-size component cap (else building-sized "person" blobs → GB of masks), and
latency is measured while energy is assumed → the measurement column is labelled
`estimated`. `experiments/real_segmentation_pilot/STATUS.md` is the current state;
the old "no real pilot" claims there are superseded.

**V3 P3 policy skyline** (`aerointentbench/v2/skyline.py`,
`aerointentbench/policies/budget_planner.py`, `docs/v3_design.md` §P3): the skyline
is a GT-AWARE offline upper bound (forward DP over the closed loop, per-slot outcomes
from the mission's own renderer/executors/evaluator, Pareto pruning on
clock/energy/communication, legal actions only — privacy-forbidden configs excluded,
budget-exceeding branches cut). Every output is labelled `gt_aware: true` + "not a
policy"; NEVER present a skyline number as a policy/baseline result. target_recall
contracts only (loud error otherwise); soundness/determinism/replay-consistency
pinned by tests. `budget_planner` is a registered V1 policy (pro-rata comm pacing
with bounded burst + EMA battery-drain projection from its own observations;
stateful within one episode — composition root builds per run). Result on both P2
bases: skyline recall 1.0 (satisfiable; light + 2 remote + 2 perfectly-timed strong)
vs rule_based 0.75 vs budget_planner FAIL 0.5 — the planner's duty-cycling halves
the strong cadence and misses odd-slot lates. Both findings kept honestly: large
measurable headroom AND sophistication-does-not-auto-win; do not tune the planner
against the family. Artifacts `results/v3_skyline/` (local-only).

**V3 P2 statistical hardening** (`docs/v3_design.md` §P2): two remote-aware hard bases
(`demo_img1_remote_hard.json` / `demo_img2_remote_hard.json`) put all five constraint
axes in play — the intended 4-way pattern (light→quality, strong→battery,
remote→communication FAIL; rule_based SUCCESS via remote→local_strong→local_light,
two switches) was verified with real models on both worlds before freezing. Family
2.0: battery and late-height bands are RELATIVE to the base (families port across
bases/worlds); network sampling jitters regime boundaries ±2 s and scales link
quality ×0.75–1.3 / RTT ×0.85–1.25, but regime structure, order, packet loss, and
reachability classes are base identity and never resampled; small-target band stays
the absolute Stage-B 0.70–0.80 m. Result (seeds 0–99, both worlds): rule_based
55/100 [0.452,0.644] on img_1 and 32/100 [0.237,0.417] on img_2 vs ALL three statics
0/100 [0,0.037]; zero counterexamples; rule averaged 11.7 remote attempts with zero
failures. Honest flags kept: ~half the successes clear the battery floor by <0.01
(deliberate knife-edge); every success is exactly two switches (two-stage escalation,
not free-form adaptation); img_2's lower rate documents world sensitivity — never
quote a single cross-world number. `multi_seed_eval` auto-selects the 4-policy set
for remote bases, records comm/privacy/network_behaviour fields, and generalises the
paired comparison to N statics. Batch results live in `results/v3_remote_multiseed/`
(local-only).

**V2.4 multi-seed evaluation** (`aerointentbench/v2/scenario_family.py`,
`multi_seed_eval.py`, `docs/v2_design.md` §10.6): the hard scenario generalises to a
deterministic family — every variant is a pure function of (family version, base id,
seed), all sampled values recorded in `provenance.scenario_family`, strict JSON, no
wall-clock in identity. Rules: a sampling change is a FAMILY_VERSION bump; every policy
runs the same scenario instances (paired by construction); per-run records restate the
evaluator's outputs (never a second success computation) and exclude measured
wall-clock; Wilson CIs via the V1 helper; counterexample seeds are preserved and
surfaced, never hidden; the family must not be re-tuned to force adaptive wins (family
1.0's honest failure is documented as sensitivity evidence). Generator + statistics are
CI-safe (no torch/assets); batch execution is local-only via
`python -m aerointentbench.v2.multi_seed_eval`.

**V2.3 replay viewer** (`aerointentbench/v2/replay_export.py`, `replay_viewer.py`,
`docs/v2_design.md` §10.4): `export-replay` runs one mission (runtime snapshots on) and
writes a self-contained bundle (`manifest.json` + `events.json` + re-rendered frames +
embedded-data `index.html`; `replay_schema_version "1.0"`). Rules: snapshots are opt-in
(`record_runtime_snapshots`, default off — existing results stay byte-identical) and
`at_capture` is exactly the policy-visible `RuntimeState`, never GT; frames/scores come
from the mission's own renderer/executor/evaluator (no second matching or success
implementation — the exporter verifies its tallies against the evaluator and fails on
divergence); viewer hierarchy is mission outcome → constraints → behaviour → perception
diagnostics, with GT only behind the labelled "Debug GT" toggle; constraint status
(SAFE/AT_RISK/VIOLATED/NOT_APPLICABLE/UNKNOWN) is presentation-layer only. **Privacy is
NOT_APPLICABLE in V2** (no remote path exists); align V1/V2 privacy semantics — V2
mission success has no privacy branch — before introducing any remote executor/config.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup
.venv/bin/pytest                                            # full suite (V2 tests skip without the [v2] extras; real-model/asset tests are opt-in markers)
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # lint and format

# run the benchmark (profile is the default executor)
.venv/bin/python -m aerointentbench.run_benchmark --suite \
  --contract data/contracts/contract_001.json --policy rule_based

# select an executor: profile | replay | real_segmentation
.venv/bin/python -m aerointentbench.run_benchmark \
  --episode data/episodes/episode_001.json \
  --contract data/contracts/contract_001.json --policy rule_based --executor replay

# capture a replay set from the profile executor
.venv/bin/python -m aerointentbench.tools.record_replay \
  --episode data/episodes/episode_001.json --output data/predictions/synthetic_replay_episode_001.json

# convert external empirical data into a runnable replay bundle, then validate it
.venv/bin/python -m aerointentbench.tools.build_empirical_bundle \
  --manifest data/examples/empirical_source/manifest.json --output data/generated/example_bundle --validate
.venv/bin/python -m aerointentbench.tools.validate_empirical_bundle --bundle data/generated/example_bundle

# regenerate the pinned reference results (review the diff!)
.venv/bin/python -m tests.test_reference_suite --update
```

## Executors

`--executor` chooses the backend; `profile` is the default and existing commands are
unchanged. `replay` serves precomputed `frame × config` records from `data/predictions/`,
resolved by `episode_id`, and is the bridge to real predictions. `real_segmentation` is a
V1 stub whose construction raises. The backend that ran is recorded as `executor_id`, read
from the executor object so it cannot be mislabelled; do not add a parallel name parameter.

**Empirical mask replay** (`docs/v1_spec.md` §4.15) lets replay predictions carry real
instance masks and ground truth carry per-frame masks with track IDs; the evaluator computes
mask IoU, matches one-to-one per frame, and deduplicates finds by hidden track ID. Which
scoring path runs is decided by the **ground-truth form** (mask `frames` vs visibility
`targets`), never by the executor — `EpisodeRunner` still contains no mask/IoU logic, and all
of it lives in the human-search task. Masks stay pure-Python (no numpy/scipy): the zero-runtime-dependency
rule is not negotiable, so the matcher is a documented deterministic greedy, not Hungarian.
The masks shipped under `data/examples/empirical_replay/` are a **synthetic correctness
example**; no real model or dataset is included, and `real_segmentation` is still only a stub.

Empirical metrics keep two counting units apart and never mix them: `target_recall`
(track-level, the canonical mission metric) and `detection_precision` (detection-level), plus
the false-positive burden. **Track-level precision and F1 are not computed** — independent
per-frame masks carry no persistent predicted-track identity — and appear as `null`
(`EmpiricalQualityScores`). Do not resurrect a `target_f1` from unique-track TP over
frame-level FP; that mixed-unit value was the bug `fix/v1-empirical-metric-semantics` removed.
The precomputed path (profile, legacy replay) is untouched and still yields `QualityScores`
with `target_precision`/`target_f1`.

**The bundle builder** (`aerointentbench.tools.build_empirical_bundle` +
`validate_empirical_bundle`, `docs/v1_spec.md` §4.16) converts external data — a manifest,
ground-truth JSON, per-config prediction JSONL, and measurement CSV — into a self-contained
replay bundle the ordinary CLI runs unchanged. It is **data ingestion only**: it executes no
model and measures no hardware, so keep model/dataset logic out of it and out of the core.
Predictions may not carry a GT track id or a trusted `mask_iou`; measurements are never
inferred from a blank cell; provenance must stay honest (never label hand-authored or
estimated numbers as `measured`). Builds are deterministic except one provenance timestamp
(`--created-at` pins it). The committed `data/examples/empirical_source/` is synthetic and
tests conversion correctness only — still no real model or dataset in the repo.

**Real-model tooling lives under `experiments/` — never in the core package.** The benchmark
runtime stays zero-dependency and model-agnostic; heavy deps (torch, etc.) belong to an
experiments-specific `requirements.txt`. `experiments/real_segmentation_pilot/` is the pilot
bridge (DatasetAdapter / SegmentationModel boundary, mask resize, latency protocol, energy
provenance) feeding the bundle builder. **No real pilot has been run** — this environment has
no dataset, checkpoints, model stack, or GPU (see its `STATUS.md`); the tooling is verified
with a labelled stub only. Do not fabricate pilot results: no hand-authored masks, no invented
latency/energy, and never present a stub run or an `estimated` energy value as a real measured
result. **The V1 empirical infrastructure is complete, but no publication-quality real
dataset/model pilot is included yet.**

**Integrated-and-frozen (V1 empirical stack).** The four empirical commits are integrated into
`develop/v1` and their schemas frozen at `schema_version "1.0"` (`docs/v1_spec.md` §14.1). Two
interface rules to preserve: (1) **fallback is model-agnostic** — resolved by
`benchmark.resolve_fallback_config_id` (explicit `--fallback-config-id` → episode
`fallback_config_id` → `initial_config_id` → legacy `CFG_LOCAL_LIGHT` if present → fail); a
data root need not contain `CFG_LOCAL_LIGHT`, and do not reintroduce that assumption. (2)
**false-positive rates are named for their denominators** — `false_positives_per_processed_minute`
(examined footage, from the evaluator) and `false_positives_per_mission_minute` (wall-clock,
from the episode metrics); do not revive the ambiguous `false_positives_per_minute`.
