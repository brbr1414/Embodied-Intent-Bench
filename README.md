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

557 tests, clean under `ruff check` and `ruff format`. Known limitations are recorded in
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

### Execution backends

`--executor` selects how a chosen configuration is turned into a result. Profile is the
default; existing commands are unaffected.

| Backend | What it does | Numbers are |
|---|---|---|
| `profile` (default) | Synthesises latency, energy, and communication from a per-platform profile, plus the remote-latency network model. Quality comes from the synthetic prediction model | **synthetic** |
| `replay` | Serves precomputed `frame × config` records from `data/predictions/`, resolved by the episode's ID. The intended bridge to real predictions; records may carry real prediction **masks** | replayed |
| `real_segmentation` | V1 **stub**; selecting it fails immediately with a clear message | — |

```bash
# replay a recorded set (must declare this episode's episode_id under predictions/)
python -m aerointentbench.run_benchmark \
  --episode data/episodes/episode_001.json \
  --contract data/contracts/contract_001.json \
  --policy rule_based --executor replay \
  --output results/episode_001_replay.json
```

A record set is captured from the profile executor with
`python -m aerointentbench.tools.record_replay --episode … --output …`. The shipped
`synthetic_replay_episode_001.json` was made this way; it keeps only the policy-visible
prediction fields, so replay reproduces the profile run's **resources exactly** (latency,
energy, communication) while quality scores as all-false-positive. Replay is the seam a
real hardware capture would fill; the executor and loader do not change when it does.

The executor that produced a result is recorded in the result JSON as `executor_id`, read
from the executor object itself so it cannot disagree with what actually ran.

### Empirical mask replay

Profile mode generates quality synthetically: a per-tier probability decides whether a
target is detected and a precomputed scalar stands in for mask overlap. **Empirical mask
replay** removes the stand-in. Replay predictions carry actual instance **masks**, hidden
ground truth carries per-frame **masks with track IDs**, and the evaluator computes mask IoU
itself and matches predictions to targets **one-to-one** per frame.

The distinction from profile is not the executor alone — it is the ground truth. A stream
whose `ground_truth/` file declares per-frame `frames` of masks is scored by IoU; one
declaring visibility `targets` is scored by the precomputed scalar. Predictions never carry a
ground-truth track ID, IoU is computed from the masks and never read off the wire, and the
policy sees none of it. A result scored this way is tagged `quality_evaluation:
"empirical_mask_iou"` in its quality details, so a saved result distinguishes synthetic
profile, legacy scalar replay, and empirical mask replay.

**Two metric families, kept apart.** Empirical scoring reports two unit-consistent groups
and never divides one into the other:

| Metric | Unit | Definition |
|---|---|---|
| `target_recall` (canonical) | mission, **track-level** | unique GT tracks found / total valid tracks — deduplicated by hidden track ID |
| `detection_precision` | frame, **detection-level** | matched predictions / non-ignored predictions |
| `false_positive_detections`, `false_positives_per_minute` | frame | the false-positive burden (per minute of examined 1 fps footage) |

**Track-level precision and F1 are not reported.** They would need a prediction tied to a
persistent predicted *track* across frames, and independent per-frame masks carry no such
identity — a per-frame `prediction_id` is not a track. They are surfaced as `null` with a
stated reason (`track_level_metrics_available: false`), never as a mixed-unit number. The
canonical empirical mission-quality metric is therefore **`target_recall`**, and the example
empirical contract uses it.

A tiny, hand-verifiable example ships under `data/examples/empirical_replay/` — 8×8 frames
with three tracked people, one exact match, one partial match above threshold, one
below-threshold detection, one false positive, one missed target, and one ignore region. Run
it through the normal CLI:

```bash
python -m aerointentbench.run_benchmark \
  --data-root data/examples/empirical_replay \
  --episode data/examples/empirical_replay/episodes/episode.json \
  --contract data/examples/empirical_replay/contracts/contract.json \
  --policy rule_based --executor replay \
  --output results/empirical_example.json
# quality metric target_recall = 0.667 (2 of 3 tracks found)
# detection_precision 0.6 (3 of 5 detections matched), false_positive_detections 2,
# false_positives_per_minute 40.0; target_f1 = null (track-level, unavailable)
```

This branch adds no real model and no dataset: the masks are a **synthetic correctness
example**, and `real_segmentation` remains a stub. Empirical replay is where recorded
`frame × config` masks from a real capture would eventually plug in unchanged.

### Building a bundle from real data

You run any model **outside** the benchmark, write its outputs to files, and a builder
converts them into a self-contained replay bundle the ordinary CLI runs unchanged. The
builder executes no model and measures no hardware — it ingests, validates, and packages.

Source formats (all stdlib, no image-decoding dependency):

| Source | Format | Contains |
|---|---|---|
| Ground truth | JSON `{instances:[…]}` | per instance: `frame_id`, `track_id`, `category`, `mask` `{height,width,rows}`, optional `ignore` |
| Predictions (per config) | JSON Lines | per line: `frame_id`, `prediction_id`, `category`, `confidence`, `mask` — **never** a GT track id or `mask_iou` |
| Measurements (per config) | CSV | `frame_id, success, latency_s, compute_energy_j, upload_mb, download_mb, failure_reason` — latency/energy never inferred from a blank cell |

A versioned **manifest** ties them together and declares the mission scaffolding and honest
**provenance** (`data_origin`, per-config `prediction_provenance` / `measurement_provenance`)
so a bundle can never label a hand-authored or estimated value as measured. **Coverage** is
explicit: `strict` (default) requires a measurement for every declared `frame × config`;
`sparse` turns a missing pair into an explicit failed record — never a silent omission.

```bash
# convert sources -> a validated, self-contained bundle
python -m aerointentbench.tools.build_empirical_bundle \
  --manifest data/examples/empirical_source/manifest.json \
  --output data/generated/example_bundle --validate

# validate an existing bundle independently
python -m aerointentbench.tools.validate_empirical_bundle --bundle data/generated/example_bundle

# then run it with the ordinary benchmark command — no custom flags
python -m aerointentbench.run_benchmark \
  --data-root data/generated/example_bundle \
  --episode data/generated/example_bundle/episodes/episode.json \
  --contract data/generated/example_bundle/contracts/contract.json \
  --policy rule_based --executor replay --output results/example.json
# -> target_recall 0.667, detection_precision 0.6 (built from the committed source fixture)
```

Builds are **deterministic**: records ordered by `(frame_id, config_id)`, JSON written
sorted, and the only non-deterministic value — the build timestamp — isolated to one
provenance field (`--created-at` pins it). Every generated file's SHA-256 is recorded in
`provenance.json`. The committed source fixture under `data/examples/empirical_source/` is
**synthetic and tests conversion correctness only** — no model, no dataset.

### Measured baselines

Pooled over 50 seeds per episode (n = 150), because three episodes can only produce a
success rate of 0, 1/3, 2/3 or 1 — an interval too wide to compare policies with:

| Policy | Mission Success Rate | 95 % CI | Why it fails |
|---|---:|---|---|
| `always_local_light` | 0.7 % | [0.1, 3.7] | almost never reaches the quality threshold |
| `always_local_strong` | 64.7 % | [56.7, 71.9] | competitive; loses where remote was worth spending on |
| `always_remote_strong` | **0.0 %** | [0.0, 2.5] | second-best quality, over the communication budget **every run** |
| `rule_based` | **78.0 %** | [70.7, 83.9] | uses remote while bandwidth is high, then rations |
| `rule_based --hide-profiles` | 0.7 % | [0.1, 3.7] | cannot tell configurations apart without a quality tier |

`rule_based` beats the best static baseline by **+13.3 points (z = 2.58, p = 0.010)**.
`always_remote_strong`'s zero is structural rather than a small-sample artifact: its
communication volume is deterministic and exceeds the budget on every run.

Reproduce with:

```bash
python -m aerointentbench.run_benchmark --suite \
  --contract data/contracts/contract_001.json --policy rule_based --repeats 50
```

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
