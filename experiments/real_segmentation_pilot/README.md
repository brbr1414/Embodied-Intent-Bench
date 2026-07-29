# Real-segmentation pilot

The bridge from a real aerial dataset and real segmentation models into an AeroIntentBench
empirical replay bundle. Model execution happens **here, outside the benchmark runtime core**;
the core stays zero-dependency and model-agnostic, and bundle replay remains the primary
reproducible evaluation path.

> **Read `STATUS.md` first.** In this repository no real pilot has been run: there is no
> dataset, no checkpoints, no model stack, and no GPU. What is committed is verified tooling
> plus a stub-driven test pipeline — **no real bundle, no fabricated numbers.** This document
> describes how to run the real pilot once those assets exist.

## Pipeline

```
your DatasetAdapter        your SegmentationModel(s)
        │                            │
        ├── ground truth (masks) ────┤
        │                            ▼
        │                  run_inference  ──►  predictions_<cfg>.jsonl + measurements_<cfg>.csv
        ▼                                                     │
   build_pilot ── writes ground_truth.json + manifest.json ──┤
        │                                                     ▼
        └──────────►  build_empirical_bundle  ──►  validate_empirical_bundle
                                     │
                                     ▼
                     normal AeroIntentBench CLI (--executor replay)
```

## Choosing a dataset

The dataset must provide, or be transformable into: aerial/UAV imagery, **person instance
masks** (not boxes — do not fake masks from boxes), frame order, and persistent person **track
ids where genuinely available**. If it has masks but no track ids, record that limitation; a
target seen across frames will otherwise be counted as one find only if you supply a valid
identity mapping. **Do not infer identity from spatial proximity** without a real, validated
tracker. Candidate families to evaluate for suitability: UAV person-segmentation / aerial
pedestrian datasets with instance masks and per-track ids. Record the exact dataset, version,
and license in the analysis report; do not commit the dataset.

Write a `DatasetAdapter` (see `interfaces.py`) for your dataset's on-disk layout. It yields
ordered `Frame`s and hidden `GroundTruthInstance`s (category, mask, track_id, ignore). Keep it
in this directory — never in the benchmark core.

## Environment

The benchmark core needs none of this. Create a separate pilot environment:

```bash
python -m venv .venv-pilot
.venv-pilot/bin/pip install -e ".[dev]"
.venv-pilot/bin/pip install -r experiments/real_segmentation_pilot/requirements.txt
```

Record the exact hardware (CPU/GPU, driver, CUDA), OS, Python, and package versions in the
analysis report. Fix seeds and deterministic flags where the framework allows.

## Configurations

Implement at least two **genuinely different** `SegmentationModel`s — e.g. a smaller model or
lower input resolution (CONFIG_LOCAL_LIGHT) versus a larger model or higher resolution
(CONFIG_LOCAL_STRONG). They must be real executable differences, not two labels for one call.
No remote inference, pruning, split inference, or new power modes in this pilot.

## Latency measurement boundary

Fix one timing boundary and use it for **every** configuration. State in the report exactly
what the timed region includes (image load / preprocess / inference / postprocess / mask
resize). Warm up first (discarded). Use a monotonic clock; on a GPU, pass a clock that
synchronises the device before reading time (so launch latency is not mistaken for inference).
The tooling records per-frame latency and reports mean / median / p95 / min / max.

## Energy source

There is no silent zero. Pick one and label it honestly:

- `estimated` (the built-in fallback): `--average-power-w P --energy-source "<where P came
  from>"`; energy = P × measured time, marked `estimated`, with a **non-binding** battery
  constraint in the contract. Not suitable for final energy conclusions.
- `measured` / `externally_supplied`: pass real telemetry via `run_config(..., energy_for=...)`
  using `experiments.real_segmentation_pilot.measure.external_energy`.

Communication is local-only for this pilot: `upload_mb = download_mb = 0`.

## Commands

```bash
# 1. run each configuration over the SAME frames (adapter/model are your module:callable)
.venv-pilot/bin/python -m experiments.real_segmentation_pilot.run_inference \
  --adapter my_pkg.dataset:build_adapter --model my_pkg.models:build_light \
  --average-power-w 15 --energy-source "assumed 15 W board TDP, vendor datasheet" \
  --output data/generated/real_segmentation_pilot/sources
# ...repeat for build_strong...

# 2. build + validate the bundle from the sources (uses the existing builder)
#    (build_pilot.run_pilot does steps 1+2 together from Python; or call the builder directly)
.venv-pilot/bin/python -m aerointentbench.tools.build_empirical_bundle \
  --manifest data/generated/real_segmentation_pilot/sources/manifest.json \
  --output data/generated/real_segmentation_pilot/bundle --validate
.venv-pilot/bin/python -m aerointentbench.tools.validate_empirical_bundle \
  --bundle data/generated/real_segmentation_pilot/bundle

# 3. run the benchmark — static baselines and rule-based, same stream/contract
for policy in always_local_light always_local_strong rule_based; do
  .venv-pilot/bin/python -m aerointentbench.run_benchmark \
    --data-root data/generated/real_segmentation_pilot/bundle \
    --episode  data/generated/real_segmentation_pilot/bundle/episodes/episode.json \
    --contract data/generated/real_segmentation_pilot/bundle/contracts/contract.json \
    --policy $policy --executor replay \
    --output results/real_pilot_$policy.json
done
```

`build_pilot.run_pilot(...)` is the one-call Python API that does run + build + validate.

## Contracts

Create a small contract suite for the same stream: a quality-focused contract (higher
`target_recall`, relaxed deadline/battery) and a latency-focused contract (tighter deadline,
moderate recall). Set thresholds from the **observed** per-config pilot behaviour and document
how — do not tune them to force the adaptive policy to win. Add a resource-focused contract
only if energy provenance is defensibly `measured`/`externally_supplied`.

## Outputs

- Sources: `data/generated/real_segmentation_pilot/sources/`
- Bundle: `data/generated/real_segmentation_pilot/bundle/` (self-contained; hashes in
  `provenance.json`)
- Results: `results/real_pilot_<policy>.json`
- Analysis: fill `PILOT_ANALYSIS_TEMPLATE.md`

Do not commit datasets, weights, or large generated bundles. Small manifests, summaries, and
hashes may be committed.

## Limitations

- A small pilot (≈50–200 frames), **not** the final benchmark dataset.
- Remote inference is not included; communication is zero.
- UAV physics is still simplified (predefined path, no flight dynamics).
- Model execution is outside the benchmark runtime core; bundle replay is the reproducible
  path.
- Track-level precision / F1 are not produced (no persistent predicted-track identity), by
  design — recall is track-level, precision is detection-level (unchanged empirical metrics).
- Default energy is `estimated`; treat energy conclusions as provisional until telemetry.
