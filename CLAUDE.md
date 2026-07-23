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

- **Never commit to `main`.** `develop/v1` is the integration branch.
- Every feature branch is created from the **latest `develop/v1`**. `develop/v1` is a
  branch name containing a slash, not a namespace — there is no branch hierarchy.
- Implement only the current branch's responsibility; run the tests; commit logical,
  reviewable changes using conventional commit messages.
- **Do not merge into `develop/v1` without explicit authorisation from the owner.**

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup
.venv/bin/pytest                                            # 483 tests
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
