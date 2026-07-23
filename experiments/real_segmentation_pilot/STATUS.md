# Real-segmentation pilot — status

**No real empirical bundle has been produced. The real pilot is blocked on assets that are
not present in this environment.** This directory contains only reusable, tested tooling that
runs the real pilot once those assets are supplied. Per the pilot's scientific rule, nothing
here fabricates a result — there are no hand-authored prediction masks, no invented latency,
and no made-up energy in any committed file.

## Missing assets (the exact blockers)

| Required | State in this environment |
|---|---|
| Aerial / UAV instance-segmentation dataset (person masks, frame order, track ids if available) | **absent** — no imagery or annotations in the repo; repo policy forbids committing datasets |
| Model checkpoints | **absent** — no `.pt` / `.pth` / `.onnx` weights |
| Model stack (PyTorch / Ultralytics / ONNX Runtime) | **not installed** — the benchmark is zero-dependency; imports fail |
| GPU / accelerator | **absent** — CPU-only (arm64). The latency protocol needs device synchronisation for meaningful numbers |

Because of these, the two acceptance criteria that require *real* execution — real model
inference and measured latency on real data — cannot be met here. Fabricating them is
explicitly forbidden, so they are left undone and reported.

## What is delivered instead

The task permits, when assets are missing, implementing "only the reusable pilot tooling that
can be verified." That is what this is:

- `interfaces.py` — `DatasetAdapter` / `SegmentationModel` boundaries, in-memory + stub
  implementations for tests, and a real-backend stub that raises until weights and extras are
  supplied.
- `masks.py` — nearest-neighbour binary-mask resize onto the GT grid.
- `measure.py` — the latency protocol (warm-up, monotonic clock, per-frame samples, mean /
  median / p95 / min / max) and mandatory energy provenance.
- `convert.py`, `run_inference.py`, `build_pilot.py` — write model output into the empirical
  bundle *source* formats and hand them to the existing `build_empirical_bundle` /
  `validate_empirical_bundle`, unchanged.

`tests/test_real_segmentation_pilot.py` verifies all of the above end to end using a
**stub model** over a tiny in-memory dataset — never a real model, and it commits no bundle.

## To run the real pilot (when assets exist)

See `README.md`. In short: create a pilot environment with `requirements.txt`, write a
concrete `DatasetAdapter` for your dataset and a `SegmentationModel` for each configuration,
run `run_inference` per configuration, then `build_pilot` → the standard benchmark CLI. Fill
`PILOT_ANALYSIS_TEMPLATE.md` from the real results — **it is a template with every question
still PENDING**, not a set of answers.
