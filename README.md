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

**V1 complete.** JSON inputs → a deterministic one-second decision loop → policy → executor
→ state and evidence updates → termination → metrics JSON. Runs on CPU with **zero runtime
dependencies**; re-running produces a byte-identical result.

454 tests, clean under `ruff check` and `ruff format`. Known limitations are recorded in
[`docs/v1_spec.md`](docs/v1_spec.md) §16 rather than left implicit.

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
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

The package declares **zero runtime dependencies**; `pytest` and `ruff` are the only
development ones.

## Usage

```bash
# one episode
python -m aerointentbench.run_benchmark \
  --episode data/episodes/episode_001.json \
  --contract data/contracts/contract_001.json \
  --policy rule_based \
  --output results/episode_001_rule_based.json

# the whole shipped suite, aggregated
python -m aerointentbench.run_benchmark --suite \
  --contract data/contracts/contract_001.json \
  --policy rule_based \
  --output results/suite_rule_based.json
```

```
policy: rule_based   executor: profile
episode           success  quality       MB    batt   time s  switch
EPISODE_001          PASS    0.872      380   0.327      900       1
EPISODE_002          PASS    0.837      380   0.427      900       2
EPISODE_003          PASS    0.872      380   0.277      900       1

Mission Success Rate: 100% over 3 episode(s)
  quality 100%   deadline 100%   battery 100%   communication 100%   privacy 100%
```

Useful flags: `--hide-profiles` withholds public configuration profiles from the policy;
`--include-detail` adds the step log and raw evidence (large, and it contains
ground-truth-derived fields — do not publish a detailed result file).

### Measured baselines

| Policy | Mission Success Rate | Why it fails |
|---|---:|---|
| `always_local_light` | 0 % | never reaches the quality threshold |
| `always_local_strong` | 67 % | competitive; loses where remote was worth spending on |
| `always_remote_strong` | 0 % | highest quality of all, always over the communication budget |
| `rule_based` | 100 % | uses remote while bandwidth is high, then rations |
| `rule_based --hide-profiles` | 0 % | cannot tell configurations apart without a quality tier |

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

## Reproducibility

`tests/reference/baseline_results.json` pins every baseline's score on the shipped fixtures
and is compared on every test run. A change to a fixture, a profile, the synthetic
prediction model, the evaluator, or a policy shows up there as a diff — so the benchmark
cannot move without someone approving the move.

```bash
python -m tests.test_reference_suite --update   # regenerate, then review the diff
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
