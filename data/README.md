# Benchmark data

> **All numbers in this directory are synthetic placeholders, not empirical measurements.**
>
> Latencies, energies, transfer sizes, network traces, predictions, and ground truth are
> hand-authored fixtures chosen to exercise the benchmark loop. They are **not** measured
> on any UAV, accelerator, or network, and must never be cited as hardware results.
> Files carrying such values are named with an explicit `synthetic_` / `_example` marker.

This directory holds benchmark **specifications and episode data** — the declarative
inputs to the runner. The software that consumes them lives in `aerointentbench/`.

| Directory | Contents | Populated by |
|---|---|---|
| `contracts/` | Mission contracts: quality target and hard constraints | `feature/v1-schemas` |
| `episodes/` | Initial episode state: platform, path, trace, allowed configs, seed | `feature/v1-schemas` |
| `configs/` | Configuration catalogs: `config_id`, `model_id`, typed `strategy` | `feature/v1-schemas` |
| `platforms/` | UAV platform profiles: capacity, flight power, comms energy | `feature/v1-schemas` |
| `task_specs/` | Task definitions: evidence type, matching rule, deduplication | `feature/v1-schemas` |
| `network_traces/` | Deterministic traces: stable, degrading, disconnecting | `feature/v1-schemas` |
| `profiles/` | Per-platform configuration profiles: latency, energy, transfer sizes | `feature/v1-executors` |
| `predictions/` | Precomputed frame × config predictions for `ReplayExecutor` | `feature/v1-human-search-task` |
| `ground_truth/` | Hidden ground truth, read **only** by task evaluators | `feature/v1-human-search-task` |

## Rules

- Every file carries `"schema_version": "1.0"`. V1 supports that value and no other;
  anything else must fail loading with a clear validation error.
- `ground_truth/` is hidden state. Nothing under `aerointentbench/policies/` may read it,
  and no ground-truth-derived quantity may reach a `RuntimeState`.
- Fixtures are inputs, not outputs. Benchmark results are written to `results/`, which is
  gitignored and regenerated.
