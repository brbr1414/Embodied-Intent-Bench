# Embodied-Intent-Bench — AeroIntentBench

**AeroIntentBench** is a benchmark for evaluating **intent-conditioned, resource-aware
inference configuration selection policies** in UAV missions.

A UAV flies a predefined path. During the mission, a policy observes a structured mission
contract and the current runtime state, then selects one inference configuration from a
fixed pool. The benchmark runner executes or simulates that configuration, updates time,
battery, network usage, and accumulated mission evidence, and finally computes
mission-level metrics.

> **The core question**
> Can a policy select appropriate perception inference configurations under changing
> battery and network conditions while satisfying a mission contract?

The benchmark deliberately does **not** evaluate path planning, navigation, flight
control, or open-ended language understanding.

## Status

🚧 **V1 in progress.** This branch (`feature/v1-spec-and-scaffold`) establishes the
specification, documentation, packaging, and package/test skeleton. Domain logic lands in
the subsequent branches listed in [`docs/branching.md`](docs/branching.md); the CLI is a
stub until `feature/v1-metrics-and-cli`.

## The loop

```
JSON specifications
  -> EpisodeRunner
  -> build RuntimeState
  -> Policy.select_config(contract, state, configs) -> config_id
  -> Executor executes or replays the selected config
  -> update time, battery, communication, path progress, evidence
  -> repeat until termination
  -> Evaluator computes episode metrics
  -> aggregate results across episodes
```

**Primary metric: Mission Success Rate.** An episode succeeds only if every hard
constraint passes — quality, deadline, final battery, communication budget, and privacy.

## V1 at a glance

| | |
|---|---|
| Mission | Human search along a predefined UAV path, 900 s over 4500 m |
| Task | `HUMAN_SEARCH_SEGMENTATION` (instance mask set, target F1) |
| Decision interval | 1 second (900 decisions per episode) |
| Action | One configuration ID from the allowed pool |
| Configurations | Local and remote; typed `placement` / `precision` / `input_compression` |
| Runtime variables | Battery, bandwidth, RTT, packet loss, remaining deadline, cumulative communication, path progress, evidence |
| Execution | Deterministic profile-driven simulation or replay of precomputed predictions |
| Baselines | Always-local-light, always-local-strong, always-remote-strong, rule-based |

Requires **Python 3.11+ and nothing else** — no GPU, no ML frameworks, no network access.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The package declares **zero runtime dependencies**; `pytest` is the only development
dependency.

## Usage (target interface — not yet runnable)

```bash
python -m aerointentbench.run_benchmark \
  --episode data/episodes/episode_001.json \
  --contract data/contracts/contract_001.json \
  --policy rule_based \
  --output results/episode_001_rule_based.json
```

A suite mode will run a fixture set and emit aggregate metrics. Both arrive in
`feature/v1-metrics-and-cli`.

## Repository layout

```
aerointentbench/
├── schemas/     typed models + JSON loading and validation
├── simulator/   episode runner, state, battery, network, termination, action validation
├── executor/    profile / replay / real-model execution backends
├── policies/    policy protocol and baseline policies
├── tasks/       task plugins (V1: human_search_segmentation)
├── metrics/     per-episode and aggregate metrics
└── run_benchmark.py   CLI and composition root

data/            benchmark specifications and fixtures (all synthetic)
docs/            specification, architecture, branching
tests/           pytest suite
```

## Documentation

- [`docs/v1_spec.md`](docs/v1_spec.md) — normative V1 scope, schemas, semantics, metrics
- [`docs/architecture.md`](docs/architecture.md) — module boundaries and interface rules
- [`docs/branching.md`](docs/branching.md) — branch workflow and V1 sequence
- [`CLAUDE.md`](CLAUDE.md) — conventions and constraints for automated contributors

## ⚠️ Synthetic data notice

Every profile value, network trace, prediction, and ground-truth file under `data/` is a
**hand-authored synthetic placeholder, not an empirical measurement.** They exist to
exercise the benchmark loop deterministically. Do not cite them as hardware, network, or
model performance results.

## License

MIT — see [LICENSE](LICENSE).
