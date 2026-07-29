# AeroIntentBench architecture

This document defines module boundaries and the rules that keep the benchmark core
task-agnostic. `docs/v1_spec.md` says *what* V1 does; this says *how it is allowed to be
built*.

Guiding principle:

> **Implement one real V1 path through each interface, while ensuring that adding a
> second implementation does not require rewriting the benchmark core.**

## 1. Layering and dependency direction

```
                    ┌───────────────────────────────┐
                    │           schemas             │  typed domain models + loading
                    └───────────────▲───────────────┘  (depends on nothing internal)
                                    │
    ┌───────────────┬───────────────┼───────────────┬───────────────┐
    │               │               │               │               │
┌───┴────┐    ┌─────┴────┐    ┌─────┴─────┐   ┌─────┴────┐   ┌──────┴─────┐
│policies│    │ executor │    │   tasks   │   │ metrics  │   │ simulator  │
└────────┘    └──────────┘    └───────────┘   └──────────┘   └─────┬──────┘
                                                                   │ depends only on
                                                                   │ protocols + schemas
                                                              run_benchmark (CLI)
                                                              composes concretes
```

Rules:

- **`schemas` depends on nothing internal.** Everything else speaks its types.
- **`simulator` depends on protocols, never on concrete implementations.** It must not
  import `HumanSearchSegmentationEvaluator`, `ProfileExecutor`, or any specific policy.
- **Composition happens at the edge.** `run_benchmark.py` resolves registries, constructs
  concrete components, and injects them into `EpisodeRunner` through its constructor.
- **No module-level mutable state** as a communication channel. Dependencies are
  constructor arguments.

## 2. Interfaces

Small, practical protocols. An abstraction is justified in V1 only if it has a current
use, a test, or an immediate replacement scenario.

```python
class Policy(Protocol):
    def select_config(
        self, contract: Contract, state: RuntimeState, configs: ConfigCatalog
    ) -> str: ...


class Executor(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class TaskEvaluator(Protocol):
    def evaluate(
        self, evidence: EvidenceRecord, ground_truth: GroundTruth, contract: Contract
    ) -> TaskEvaluationResult: ...


class EvidenceTracker(Protocol):
    def update(self, result: ExecutionResult) -> None: ...
    def policy_summary(self) -> EvidenceSummary: ...
    def final_record(self) -> EvidenceRecord: ...


class BatteryModel(Protocol):
    def transition(
        self, previous_state: BatteryState, usage: EnergyUsage, elapsed_time_s: float
    ) -> BatteryState: ...


class NetworkModel(Protocol):
    def observe(self, time_s: float) -> NetworkObservation: ...


class TerminationCondition(Protocol):
    def check(self, state: SimulationState, contract: Contract) -> TerminationReason | None: ...
```

V1 concrete implementations: `ProfileExecutor`, `ReplayExecutor`,
`HumanSearchEvidenceTracker`, `HumanSearchSegmentationEvaluator`, `SimpleBatteryModel`,
`TraceBasedNetworkModel`, the static and rule-based policies.

## 3. Registries

Lightweight, explicit Python registries — no entry-point systems, no dynamic plugin
discovery. Each maps a key to a factory:

```python
task = task_registry.create(contract.task_id, task_spec)
policy = policy_registry.create(policy_name, policy_config)
executor = executor_registry.create(executor_type, profiles)
```

A registry is added by the branch that gets a second implementation to register — not
before. A registry with nothing in it is indirection without a use.

## 4. Execution abstraction

The benchmark separates three things that are easy to conflate:

1. **what** configuration was selected (`Configuration`),
2. **how** it is executed (`Executor` implementation),
3. **what result** it produced (`ExecutionResult`).

`ExecutionRequest` carries: episode ID, frame ID / input reference, selected
configuration, current network observation, current time, deterministic seed.

`ExecutionResult` carries: success, `latency_s`, energy usage, communication usage,
prediction/evidence payload reference, failure reason, executor-specific metadata.

The runner must work identically whether the result came from a synthetic profile, a
precomputed prediction, a real local model, a remote server, or a future split-inference
runtime.

## 5. Policy-visible vs. hidden state

Three distinct tiers, and the boundary is enforced, not merely intended:

| Tier | Contents | Reachable by policy |
|---|---|---|
| **Policy-visible** | `Contract`, `RuntimeState`, allowed configuration descriptions | yes |
| **Simulator-internal** | full episode object, network trace, energy ledger, step log | no |
| **Hidden ground truth** | `data/ground_truth/`, match outcomes, true target counts | never |

- A policy receives an **immutable (frozen) `RuntimeState`**, never the `Episode` or the
  runner itself.
- `EvidenceTracker.policy_summary()` returns prediction-derived quantities only. Any
  ground-truth-derived value appearing there is a leak and a benchmark-invalidating bug.
- The full future network trace is simulator-internal; the policy sees only the current
  observation.
- Public configuration **profile** visibility is an explicit benchmark-mode setting passed
  to the policy, never an accidental import.

Future versions may change what is visible, which is exactly why the boundary is a named
type rather than a convention.

## 6. Records, not simulator internals

Four separate concepts, deliberately not collapsed:

1. **Step-level execution log** — timestamp, frame ID, runtime state snapshot, selected
   configuration, validated configuration, execution success, latency, energy usage,
   communication usage, evidence update reference, violations/failures.
2. **Final episode record** — the whole run, JSON-serialisable.
3. **Task-specific evaluation result** — standardised `TaskEvaluationResult`.
4. **Generic constraint evaluation and aggregate metrics** — computed *from* records.

Evaluators and metrics read records. They never reach into live simulator state. A metric
that was not computed at run time can be added later without rerunning anything, provided
the step log already carried the data — which is why the step log is generous.

## 7. Anti-patterns (rejected in review)

- `EpisodeRunner` importing a concrete task evaluator.
- Mask/IoU logic anywhere outside `tasks/human_search_segmentation/`.
- Policy code hardcoding fixture config IDs as behaviour.
- Metric aggregation naming a task-specific field such as `target_f1`.
- Executors with `if/elif` chains over tasks or model IDs.
- Deriving behaviour by parsing `config_id` strings.
- Strategy-specific fields scattered through the simulator instead of living in typed
  configuration metadata.

Preferred throughout: composition over inheritance, small protocols over deep hierarchies,
explicit factories over hidden globals, typed records over loose dicts, constructor
injection over module state, deterministic pure functions where practical.

## 8. Extensibility acceptance criteria

Each of the following must be possible **without modifying `EpisodeRunner` core logic**.
Adding a registry entry, a config file, or a new module is fine; adding task-specific
`if/elif` to the runner is not.

1. Add an object-detection task evaluator.
2. Add a new policy implementation.
3. Replace `ProfileExecutor` with `ReplayExecutor`.
4. Replace `SimpleBatteryModel` with a trace-based battery model.
5. Add a new configuration strategy field.
6. Add an optional task-specific metric.
7. Add a new network trace.
8. Add a future external simulator adapter.

At least one test must *demonstrate* replaceability. As of `feature/v1-metrics-and-cli`,
criteria 1–4 are exercised: a second task registered without touching the shipped registry,
a policy defined as a bare class conforming to the protocol, an executor swapped through the
real runner, and a replacement battery model injected into `StateManager`. Two tests also
check the core rule mechanically rather than trusting it — `EpisodeRunner` is parsed to
confirm it imports no task or policy module and never reads `task_id`, and
`episode_metrics.py` is parsed to confirm it names no task-specific metric.

## 9. Anticipated but not implemented

These shape interfaces only. **Do not implement them without an explicit request.**

- Tasks: object detection, semantic/instance segmentation, classification, tracking, depth.
- Missions: search and rescue, infrastructure inspection, wildfire monitoring, mapping,
  target tracking.
- Strategies: pruning, quantization, early exit, split inference, adaptive compression,
  frame skipping, dynamic resolution, power-mode selection.
- Platforms: other UAV hardware, edge servers, heterogeneous accelerators, other embodied
  platforms.
- Environment models: realistic battery models, recorded power traces, network replay,
  Gazebo/PX4/ROS 2 adapters.
- Policies: profile-greedy, MPC, contextual bandit, RL, externally submitted policies.
- Additional metrics and contract fields; schema evolution across benchmark versions.

## 10. Where things go

| Concern | Location |
|---|---|
| Typed models, JSON loading, schema-version gate | `aerointentbench/schemas/` |
| Decision loop, state assembly, battery, network, termination, action validation | `aerointentbench/simulator/` |
| Execution backends | `aerointentbench/executor/` |
| Policies + policy registry | `aerointentbench/policies/` |
| Task evidence trackers and evaluators + task registry | `aerointentbench/tasks/` |
| Generic constraint and aggregate metrics | `aerointentbench/metrics/` |
| Composition root | `aerointentbench/benchmark.py` |
| CLI | `aerointentbench/run_benchmark.py` |
| Component registries | `aerointentbench/registry.py` + one per package |
| Specifications and fixtures | `data/` |
| Schema migrations (future) | `aerointentbench/schemas/loading.py` |
