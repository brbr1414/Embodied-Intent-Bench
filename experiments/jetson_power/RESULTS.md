# Jetson power sweep — measured results (2026-08-03)

Real hardware measurement of the six V3x model-zoo torchvision segmentation
checkpoints on a Jetson AGX Orin, across four power modes. This is a standalone
measurement record: **no benchmark scenario, configured value, or documented result
was changed based on these numbers.** (A proposal to re-derive the model-zoo
scenario's configured latency/energy ratios from this data is pending an explicit
owner decision — see "Relation to the benchmark" below.)

## Setup

| | |
|---|---|
| Device | Jetson AGX Orin 64GB, JetPack R36 (rev 4.0) |
| Stack | Python 3, torch 2.8.0 cu126, torchvision official DEFAULT weights |
| Input | `1x3x384x512` uniform random, seed 0 (**512×384** — see caveat 1) |
| Power | INA3221 module rails (VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0) sampled ~20 Hz in a background thread |
| Protocol | per cell: 10 s idle baseline → 5 warmup inferences → CUDA-synchronised timed inferences until ≥30 iters AND ≥10 s wall |
| Grid | 4 power modes × 6 models × 3 repeats = 72 cells, all completed |
| Governor | default DVFS; `jetson_clocks` NOT applied |

Energy per inference is reported two ways: **total** = mean run power × wall /
iterations; **marginal** = the same minus the same-invocation idle mean. Raw 20 Hz
samples and per-inference latencies are preserved per cell so aggregation can be
redone.

Scripts: `measure.py` (one cell per invocation, ran on the device) and
`aggregate.py` (restates the measurement files; nothing re-measured). Raw records
and the machine-readable summary live in `results/jetson_power/` (gitignored,
local-only, per repo convention): `raw/<MODE>__<model>__rep<N>.json` × 72 and
`jetson_power_summary.json`.

## Results (3 repeats per cell; latency ±std across repeats; p95 pooled)

| mode | model | latency ms (±std) | p95 ms | idle W | run W | E_total J/inf (±) | E_marginal J/inf (±) |
|---|---|---|---|---|---|---|---|
| MODE_15W | lraspp_mobilenet_v3_large | 28.2 ±0.1 | 28.5 | 6.33 | 10.48 | 0.296 ±0.002 | 0.117 ±0.002 |
| MODE_15W | deeplabv3_mobilenet_v3_large | 36.4 ±0.4 | 37.2 | 6.32 | 10.92 | 0.398 ±0.007 | 0.168 ±0.005 |
| MODE_15W | fcn_resnet50 | 104.0 ±0.1 | 104.3 | 6.45 | 16.87 | 1.755 ±0.004 | 1.084 ±0.029 |
| MODE_15W | deeplabv3_resnet50 | 117.8 ±0.1 | 118.0 | 6.32 | 16.33 | 1.923 ±0.007 | 1.179 ±0.010 |
| MODE_15W | fcn_resnet101 | 165.4 ±0.0 | 165.7 | 6.34 | 17.09 | 2.828 ±0.008 | 1.780 ±0.013 |
| MODE_15W | deeplabv3_resnet101 | 179.1 ±0.0 | 179.4 | 6.33 | 16.92 | 3.030 ±0.010 | 1.896 ±0.016 |
| MODE_30W | lraspp_mobilenet_v3_large | 18.8 ±0.2 | 19.2 | 6.69 | 12.44 | 0.233 ±0.001 | 0.108 ±0.003 |
| MODE_30W | deeplabv3_mobilenet_v3_large | 23.7 ±0.1 | 24.8 | 6.66 | 13.59 | 0.322 ±0.001 | 0.164 ±0.000 |
| MODE_30W | fcn_resnet50 | 73.5 ±0.1 | 75.3 | 6.66 | 22.12 | 1.626 ±0.003 | 1.136 ±0.002 |
| MODE_30W | deeplabv3_resnet50 | 81.6 ±0.7 | 84.2 | 6.68 | 21.78 | 1.777 ±0.004 | 1.232 ±0.008 |
| MODE_30W | fcn_resnet101 | 111.6 ±0.2 | 113.6 | 6.67 | 23.35 | 2.605 ±0.007 | 1.861 ±0.007 |
| MODE_30W | deeplabv3_resnet101 | 120.5 ±0.1 | 122.2 | 6.73 | 22.90 | 2.759 ±0.006 | 1.948 ±0.007 |
| MODE_50W | lraspp_mobilenet_v3_large | 20.8 ±0.1 | 21.3 | 6.72 | 12.12 | 0.252 ±0.001 | 0.112 ±0.001 |
| MODE_50W | deeplabv3_mobilenet_v3_large | 25.8 ±0.2 | 27.3 | 6.70 | 13.09 | 0.338 ±0.002 | 0.165 ±0.001 |
| MODE_50W | fcn_resnet50 | 58.5 ±0.6 | 62.0 | 6.78 | 26.72 | 1.563 ±0.009 | 1.166 ±0.004 |
| MODE_50W | deeplabv3_resnet50 | 65.1 ±0.5 | 68.9 | 6.79 | 26.40 | 1.718 ±0.007 | 1.276 ±0.004 |
| MODE_50W | fcn_resnet101 | 87.6 ±0.2 | 91.2 | 6.79 | 28.88 | 2.530 ±0.004 | 1.935 ±0.003 |
| MODE_50W | deeplabv3_resnet101 | 95.1 ±0.7 | 99.8 | 6.79 | 28.34 | 2.694 ±0.006 | 2.048 ±0.002 |
| MAXN | lraspp_mobilenet_v3_large | 14.6 ±0.1 | 14.9 | 6.74 | 15.49 | 0.226 ±0.001 | 0.127 ±0.001 |
| MAXN | deeplabv3_mobilenet_v3_large | 18.4 ±0.1 | 19.5 | 6.74 | 17.31 | 0.318 ±0.002 | 0.194 ±0.001 |
| MAXN | fcn_resnet50 | 44.5 ±0.7 | 47.6 | 6.76 | 40.46 | 1.801 ±0.012 | 1.500 ±0.006 |
| MAXN | deeplabv3_resnet50 | 49.5 ±1.2 | 53.1 | 6.83 | 40.29 | 1.994 ±0.019 | 1.656 ±0.010 |
| MAXN | fcn_resnet101 | 66.6 ±0.4 | 69.3 | 6.82 | 43.41 | 2.891 ±0.011 | 2.437 ±0.008 |
| MAXN | deeplabv3_resnet101 | 71.0 ±0.2 | 73.0 | 6.83 | 43.37 | 3.080 ±0.010 | 2.595 ±0.010 |

## Findings

1. **Energy per inference is nearly power-mode-invariant for the large models.**
   deeplabv3_resnet101 costs 3.03 J (15W), 2.76 J (30W), 2.69 J (50W), 3.08 J
   (MAXN) per inference. Capping the mode trades latency (2.5× slower at 15W than
   MAXN) and peak power (17 W vs 43 W run) but barely changes joules per frame;
   MODE_30W/50W are the energy-optimal points for the ResNet models.
2. **MODE_50W is slower than MODE_30W for the MobileNet models** (20.8 vs 18.8 ms
   on LRASPP). This is a default-DVFS-governor effect; `jetson_clocks` was
   deliberately not applied, and the anomaly is reported as observed rather than
   tuned away.
3. **Measured model-to-model spreads (MODE_15W, relative to LRASPP):** latency
   1 / 1.29 / 3.69 / 4.18 / 5.87 / 6.35; total energy 1 / 1.34 / 5.93 / 6.50 /
   9.55 / 10.2. For comparison, the model-zoo scenario's *configured* values span
   5.5× in latency (close to measured) but **90×** in energy (10 → 900 J/call),
   i.e. the configured battery penalty of the largest model exaggerates the
   measured ratio by roughly 9×. The scenario values are deliberate mission-scale
   stress settings and are labelled as such; this row exists so the gap is on
   record, not to imply the scenario claims hardware realism.

## Caveats (read before citing any number)

1. **Input size 512×384 only.** The model-zoo scenario runs the ResNet-family
   configs at 768×576; these measurements do not directly transfer to that size.
   A follow-up sweep at 768×576 (3 models, no mode change needed) is required
   before grounding those configs' ratios in this data.
2. **Module rails, not wall power.** INA3221 VDD_GPU_SOC + VDD_CPU_CV +
   VIN_SYS_5V0 excludes some board consumers; treat absolute watts as the module
   envelope, not total system draw.
3. **Default DVFS, no `jetson_clocks`**, single uniform-random input, batch 1,
   fp32. Numbers characterise this configuration only.
4. These are **bench measurements of inference on a devkit**, not UAV flight
   measurements: no radio, no flight load, no thermal soak. They must never be
   presented as mission energy or as V2/V3 scenario validation.

## Relation to the benchmark

~~Nothing in `data/`, `docs/`, or any scenario was modified. The pending proposal
(owner decision required): re-derive `demo_img1_model_zoo.json` configured
latency/energy values.~~ **Decided and executed 2026-08-06** (owner approval): the
measured values became two NEW hardware-profile scenario variants
(`demo_img1_model_zoo_orin.json` / `_xavier.json`, generated by
`make_hw_profiles.py` in this directory) — measured absolute local latency +
marginal energy, idle floor folded into `flight_power_w`, measured raw-RGB upload
sizes; the original stress scenario and its documented results are untouched.
Design rules, verified outcomes, and honest findings: `docs/v3x_extensions.md` §4.

## Board state

The shared Jetson (`rajat@128.195.55.171`) was left in **MODE_15W** after the
sweep (its default is MODE_30W, ID 2). ~~Restoring it requires `sudo nvpmodel -m 2`
and another reboot; deferred until it is clear no follow-up sweep is wanted.~~
**Resolved 2026-08-05:** the board was found rebooted and back in its default
MODE_30W (rebooted by another user of the shared board); no restore action remains.

---

# Session 1 (2026-08-05, MODE_30W) — resolution scaling, switch costs, client-side remote path

Covers plan items A1 (768×576 sweep), A2/A3 (model-switch and cold-start costs),
A5 (preprocessing), and B1/B2/B3 (client side of the remote path). Same protocol
and scripts as above unless stated (`measure.py` with the resolution argument,
`measure_switch.py`, `measure_client.py`). Raw records:
`results/jetson_power/session1/` (gitignored, local-only).

**Environment deltas vs the original sweep — read before comparing numbers.**

1. **MODE_30W, not MODE_15W.** The board had been rebooted by another user and was
   back in its default MODE_30W; the owner chose to run session 1 at 30W rather
   than force another reboot. The plan's original A1 assumed 15W; a 15W 768×576
   sweep was NOT measured. Both resolutions are fully covered at 30W, so
   within-mode ratios are still well-defined.
2. **Idle floor 8.10–8.11 W vs 6.66–6.73 W in the original 30W sweep** — three
   `jtop` instances (another user's) were running throughout session 1. `E_total`
   is inflated accordingly; **`E_marginal` is the comparable column.**
3. The root filesystem was 100 % full (another user's 31 GB; not touched).
   Checkpoints were served from the pre-existing default cache
   (`~/.cache/torch/hub`); nothing was downloaded.

## A1 — 768×576 sweep (MODE_30W, 3 repeats, same protocol)

| mode | model | input | latency ms (±std) | p95 ms | idle W | run W | E_total J/inf (±) | E_marginal J/inf (±) |
|---|---|---|---|---|---|---|---|---|
| MODE_30W | fcn_resnet50 | 768x576 | 157.7 ±0.2 | 158.0 | 8.10 | 23.48 | 3.703 ±0.012 | 2.425 ±0.012 |
| MODE_30W | deeplabv3_resnet50 | 768x576 | 180.0 ±0.0 | 180.1 | 8.11 | 22.73 | 4.091 ±0.006 | 2.632 ±0.008 |
| MODE_30W | deeplabv3_resnet101 | 768x576 | 271.2 ±0.0 | 271.3 | 8.11 | 23.41 | 6.349 ±0.007 | 4.151 ±0.006 |

**Finding — cost scales with pixel area.** 768×576 has 2.25× the pixels of
512×384; measured latency is ×2.15 / ×2.21 / ×2.25 and marginal energy ×2.13 /
×2.14 / ×2.13 for the three models. No super-linear blow-up, no cache cliff.

**Zoo-relevant ratio (30W, at the resolutions the zoo scenario actually runs):**
LRASPP@512×384 → DeepLabV3-R101@768×576 spans ×14.4 in latency (18.8 → 271.2 ms)
and ×27 in total / ×38 in marginal energy per inference (0.233 → 6.349 J total,
0.108 → 4.151 J marginal). The zoo scenario's configured spread is 5.5× latency /
90× energy: the latency spread is understated ~2.6× and the energy spread
overstated ~2.4–3.3× once resolution is accounted for — much closer than the
same-resolution comparison (10×) suggested. Ratio-grounding remains a pending
owner decision; these are the numbers it would use (or a 15W rerun if 15W ratios
are preferred).

## A2/A3 — model-switch cost, cold start, resident set (MODE_30W, 512×384 input)

Fresh process per model, checkpoints already on disk (download excluded; torch
import is page-cache-warm at ~2.05 s — a truly cold boot would pay more):

| model | construct+load s | to device s | first inference ms | steady ms | cold total s* |
|---|---|---|---|---|---|
| lraspp_mobilenet_v3_large | 0.27 | 0.03 | 790 | 18.5 | 3.25 |
| deeplabv3_mobilenet_v3_large | 0.44 | 0.05 | 465 | 22.3 | 3.11 |
| fcn_resnet50 | 1.16 | 0.09 | 370 | 69.7 | 3.75 |
| deeplabv3_resnet50 | 1.30 | 0.09 | 358 | 78.8 | 3.90 |
| fcn_resnet101 | 1.91 | 0.12 | 350 | 111.1 | 4.53 |
| deeplabv3_resnet101 | 2.06 | 0.14 | 377 | 120.1 | 4.71 |

\* torch import (2.05 s) + CUDA init (0.10 s) + load + transfer + first inference.

All six models resident in one process (unified memory): cumulative CUDA
allocated 804 MB / reserved 980 MB, process RSS 1.13 GB — the whole zoo fits
trivially in 64 GB. Round-robin inference across all six (one call per model per
cycle) equals dedicated steady-state latency within noise (e.g. LRASPP 18.9 vs
18.2 ms; R101 120.3 vs 120.3 ms): **switching between already-resident models is
free at inference time.**

**Findings.** (a) A config switch to a *non-resident* model costs ~0.3–2.2 s of
load plus a 350–790 ms first inference — a real but bounded penalty the benchmark
currently charges nothing for; keeping the whole zoo resident removes it entirely
at ~1 GB of memory. (b) The first inference after load is 3–43× steady state
(kernel/algorithm selection), largest *relatively* for the lightest model
(LRASPP: 790 ms vs 18.5 ms steady). This is the number a cold local fallback
would actually pay (A3), and it gives `sticky_escalation`'s dwell hysteresis a
hardware rationale only in the non-resident deployment model.

## B1 — upload encode time and real Size_up (CPU, Pillow software codecs)

Synthetic content: `structured` = seeded compressible pattern (stands in for real
imagery), `noise` = uniform random (incompressible upper bound). Real aerial
frames lie between, typically nearer `structured`.

| content | resolution | format | size MB | encode ms | decode ms |
|---|---|---|---|---|---|
| structured | 512x384 | jpeg_q85 | 0.028 | 1.6 | 1.9 |
| structured | 512x384 | jpeg_q95 | 0.066 | 1.9 | 2.6 |
| structured | 512x384 | png | 0.341 | 50.3 | 6.8 |
| structured | 512x384 | raw | 0.590 | 0.05 | — |
| noise | 512x384 | jpeg_q85 | 0.149 | 2.5 | 3.9 |
| noise | 512x384 | jpeg_q95 | 0.231 | 2.8 | 4.5 |
| structured | 768x576 | jpeg_q85 | 0.054 | 3.4 | 4.0 |
| structured | 768x576 | jpeg_q95 | 0.137 | 4.1 | 5.5 |
| structured | 768x576 | png | 0.764 | 121.9 | 14.1 |
| structured | 768x576 | raw | 1.327 | 0.11 | — |
| noise | 768x576 | jpeg_q85 | 0.335 | 5.6 | 8.6 |
| noise | 768x576 | jpeg_q95 | 0.520 | 6.3 | 10.0 |

**Finding.** The configured `communication_mb_per_call` of 2.0 MB is ~1.5× even a
*raw* 768×576 frame and 15–70× a JPEG of it (q95–q85, structured content); the
0.6 MB config matches raw 512×384 exactly and is ~9–21× a JPEG. `t_encode` is
1.6–6.3 ms for JPEG — negligible against network transfer at mission bandwidths.
Whether the benchmark may assume JPEG at all is entangled with the privacy
model's "remote raw-RGB" premise — still an explicit pending decision; these
numbers just quantify what each choice costs.

## B2 — mask download: wire format size and decode time

Synthetic binary masks (blobs; `pos` = positive-pixel fraction):

| resolution | mask | format | size | gzip | encode ms | decode ms |
|---|---|---|---|---|---|---|
| 512x384 | sparse (0.17 %) | dense_json | 590,592 B | 1,212 B | 20.5 | 25.6 |
| 512x384 | sparse | rle_json | 354 B | 138 B | 0.5 | 0.7 |
| 512x384 | sparse | png_mask | 459 B | — | 2.3 | 0.9 |
| 512x384 | moderate (7.0 %) | dense_json | 590,592 B | 2,001 B | 20.7 | 25.1 |
| 512x384 | moderate | rle_json | 3,127 B | 806 B | 0.6 | 0.8 |
| 512x384 | moderate | png_mask | 1,968 B | — | 3.1 | 0.9 |
| 768x576 | sparse (0.07 %) | dense_json | 1,328,256 B | 2,255 B | 45.2 | 56.3 |
| 768x576 | sparse | rle_json | 357 B | 141 B | 1.0 | 1.6 |
| 768x576 | sparse | png_mask | 701 B | — | 5.0 | 2.0 |
| 768x576 | moderate (3.3 %) | dense_json | 1,328,256 B | 3,137 B | 44.8 | 56.6 |
| 768x576 | moderate | rle_json | 3,276 B | 819 B | 1.1 | 1.7 |
| 768x576 | moderate | png_mask | 2,348 B | — | 6.3 | 2.0 |

**Finding.** Dense JSON — the P4 wire format that blew up to multi-GB bundles —
is 180–3,700× larger than RLE and ~30× slower to decode. RLE-JSON and PNG masks
are both sub-4 KB and sub-2 ms at these densities; either is a sound wire format
for the future real server. The configured `download_mb_per_call` (0.6 MB)
coincidentally equals dense-JSON at 512×384; under RLE it would be ~0.0004 MB.

## B3 — protocol (de)serialization

A request dict matching `InferenceRequest.to_wire()` (protocol 1.0) is 353 B and
round-trips (json encode+decode) in ~15 µs; a representative response envelope
(V3 P1 has no frozen response wire schema) is 338 B / ~15 µs. **Protocol overhead
is negligible**; the Model A `request_encoding_s` slot is effectively the payload
encode time (B1), not the envelope.

## A5 — preprocessing (1920×1080 synthetic capture → normalized CUDA tensor)

| target | resize ms | to-tensor ms | normalize ms | H2D copy ms | total ms |
|---|---|---|---|---|---|
| 512x384 | 11.3 | 0.3 | 0.4 | 0.4 | 12.3 |
| 768x576 | 13.6 | 0.6 | 0.8 | 0.7 | 15.6 |

**Finding.** Preprocessing is dominated by the CPU (PIL bilinear) resize and
totals 12–16 ms — **~65 % of a LRASPP inference (18.8 ms at 30W)** and therefore
not negligible for light configs, though only ~5 % of a 768×576 R101 inference.
The benchmark currently charges nothing for it.

## Session 1 caveats (Orin)

1. All B1/B2 payload sizes are **synthetic-content** measurements (structured
   pattern / uniform noise / blob masks) on Pillow **software** codecs (no
   NVJPEG); JPEG and RLE sizes are content-dependent. Never quote them as real
   aerial payload sizes — the structured/noise pair brackets, not replaces, real
   content.
2. Latency-only for A2/A3/A5/B1–B3 (no power sampling); A1 energy caveats as in
   the main sweep, plus the elevated idle floor (jtop) — prefer `E_marginal`.
3. Cold-start numbers exclude checkpoint download and benefit from a warm page
   cache (torch import 2.05 s); a cold-boot UAV would pay more.
4. MODE_30W only; the 15W 768×576 cells do not exist.

---

# Second board: Jetson AGX Xavier (2026-08-05) — full campaign redo

The same measurement program repeated on a second, older-generation shared board:
**AGX Xavier 32GB** (`rajat@128.195.55.180`, Volta GPU, L4T R35.5.0 / JetPack 5.1.3,
Python 3.8, NVIDIA torch 2.1.0a0, torchvision 0.16.2 installed `--no-deps` from the
PyPI aarch64 wheel — its compiled ops don't load against the NVIDIA torch, which
only disables `torchvision.io`, unused here). Same scripts, protocol, and synthetic
inputs as the Orin campaign; rails and device string are discovered at runtime
(`measure.py` now sums **all** labelled INA3221 rails — on Xavier two chips:
GPU+CPU+SOC and CV+VDDRQ+SYS5V; rail read permissions required a one-time
`sudo chmod a+r`, which reverts on reboot). Raw records:
`results/jetson_power_xavier/raw/` + `jetson_power_summary.json` (gitignored,
local-only). Keep the two boards' summaries in separate directories — the
aggregator must never pool across boards.

**Board-capability differences found (both benchmark-relevant):**

1. **nvpmodel switches LIVE (no reboot) between MODE_15W / MODE_30W_ALL / MAXN**
   — unlike the Orin, where every mode change required a reboot. In-mission
   power-mode adaptation is *hardware-possible* on Xavier for those modes.
   **MODE_10W is the exception**: nvpmodel demands a reboot
   ("Reboot required for changing to this power mode"), so the 10W column does
   not exist (owner-run reboot needed if ever wanted).
2. **The eMMC root is 28 GB with <500 MB free, but `/images` is a 503 GB NVMe
   (422 GB free).** The sweep ran with the download-measure-delete strategy
   (one checkpoint on disk at a time); afterwards `/images/aerobench/torch_home`
   (rajat-owned subdir; the volume also hosts Docker and other users' data) took
   all six checkpoints (792 MB), which enabled the all-6-resident test and kills
   future re-download cost.

## Sweep (3 modes × 6 models × 3 reps at 512×384; +768×576 and cold per-model extras)

| mode | model | input | latency ms (±std) | p95 ms | idle W | run W | E_total J/inf (±) | E_marginal J/inf (±) |
|---|---|---|---|---|---|---|---|---|
| MODE_15W | lraspp_mobilenet_v3_large | 512x384 | 27.3 ±0.5 | 30.5 | 5.96 | 11.47 | 0.313 ±0.004 | 0.150 ±0.002 |
| MODE_15W | deeplabv3_mobilenet_v3_large | 512x384 | 54.2 ±0.3 | 56.1 | 5.92 | 12.38 | 0.671 ±0.002 | 0.350 ±0.000 |
| MODE_15W | fcn_resnet50 | 512x384 | 319.8 ±0.0 | 320.1 | 5.92 | 14.68 | 4.694 ±0.022 | 2.800 ±0.023 |
| MODE_15W | deeplabv3_resnet50 | 512x384 | 496.2 ±0.2 | 497.8 | 5.92 | 14.40 | 7.145 ±0.032 | 4.207 ±0.031 |
| MODE_15W | fcn_resnet101 | 512x384 | 535.4 ±0.0 | 535.7 | 6.02 | 14.35 | 7.684 ±0.022 | 4.459 ±0.089 |
| MODE_15W | deeplabv3_resnet101 | 512x384 | 712.2 ±0.2 | 714.6 | 6.05 | 14.25 | 10.151 ±0.024 | 5.839 ±0.155 |
| MODE_30W_ALL | lraspp_mobilenet_v3_large | 512x384 | 22.1 ±0.7 | 25.2 | 5.92 | 13.11 | 0.290 ±0.004 | 0.159 ±0.000 |
| MODE_30W_ALL | deeplabv3_mobilenet_v3_large | 512x384 | 44.4 ±0.2 | 47.1 | 5.94 | 14.27 | 0.634 ±0.003 | 0.370 ±0.003 |
| MODE_30W_ALL | fcn_resnet50 | 512x384 | 267.9 ±63.5* | 242.4 | 5.93 | 18.19 | 4.836 ±0.886 | 3.247 ±0.510 |
| MODE_30W_ALL | fcn_resnet50 | 768x576 | 531.8 ±1.3 | 531.7 | 5.93 | 18.67 | 9.931 ±0.025 | 6.777 ±0.028 |
| MODE_30W_ALL | deeplabv3_resnet50 | 512x384 | 373.2 ±0.0 | 374.2 | 5.93 | 17.94 | 6.693 ±0.008 | 4.479 ±0.009 |
| MODE_30W_ALL | deeplabv3_resnet50 | 768x576 | 735.5 ±0.1 | 736.5 | 5.93 | 17.77 | 13.072 ±0.010 | 8.709 ±0.010 |
| MODE_30W_ALL | fcn_resnet101 | 512x384 | 404.8 ±0.0 | 405.2 | 5.93 | 18.05 | 7.305 ±0.008 | 4.904 ±0.007 |
| MODE_30W_ALL | deeplabv3_resnet101 | 512x384 | 535.9 ±0.1 | 536.8 | 5.93 | 17.74 | 9.510 ±0.016 | 6.331 ±0.015 |
| MODE_30W_ALL | deeplabv3_resnet101 | 768x576 | 1086.1 ±0.3 | 1087.1 | 5.93 | 17.75 | 19.276 ±0.005 | 12.832 ±0.004 |
| MAXN | lraspp_mobilenet_v3_large | 512x384 | 30.0 ±40.1* | 15.3 | 8.36 | 21.59 | 0.417 ±0.206 | 0.182 ±0.089 |
| MAXN | deeplabv3_mobilenet_v3_large | 512x384 | 29.0 ±0.1 | 31.6 | 8.30 | 26.95 | 0.783 ±0.002 | 0.541 ±0.001 |
| MAXN | fcn_resnet50 | 512x384 | 176.2 ±0.1 | 176.7 | 8.37 | 32.65 | 5.754 ±0.036 | 4.279 ±0.042 |
| MAXN | deeplabv3_resnet50 | 512x384 | 268.4 ±20.6 | 257.2 | 8.37 | 32.09 | 8.596 ±0.329 | 6.351 ±0.170 |
| MAXN | fcn_resnet101 | 512x384 | 293.3 ±5.5 | 290.9 | 8.28 | 31.72 | 9.299 ±0.109 | 6.872 ±0.151 |
| MAXN | deeplabv3_resnet101 | 512x384 | 370.8 ±0.2 | 371.3 | 8.34 | 32.84 | 12.176 ±0.013 | 9.083 ±0.009 |

\* High-variance cells are a **mode-switch DVFS transient, kept, not tuned away**:
the first repeat measured right after an nvpmodel switch can run far slower
(lraspp@MAXN rep1 ≈ 112 ms) before clocks settle. Three extra repeats added later
under settled clocks are tight and give the steady values: **fcn_resnet50
@30W_ALL = 242.0 ±0.1 ms / 4.47 J**, **lraspp@MAXN = 13.2–14.2 ms / ~0.33 J**
(pooled p95 in the table already reflects the steady population). All repeats are
preserved in the raw records.

## Findings (Xavier)

1. **The ResNet family is 3–4× slower and ~3× more energy-hungry than on Orin**
   (dlv3_r101@15W: 712 ms / 10.15 J total vs Orin 179 ms / 3.03 J), while the
   MobileNet models are only ~1.2–1.5× slower. On Xavier the zoo's strong tier at
   its actual 768×576 resolution costs **1.086 s / 19.3 J total per inference** —
   it cannot even sustain the benchmark's 1 Hz decision cadence (>1 s per frame).
2. **Zoo-resolution spread at MODE_30W_ALL** (LRASPP@512×384 → dlv3_r101@768×576):
   **×49 latency / ×81 marginal energy** (22.1→1086.1 ms; 0.159→12.832 J) vs the
   zoo scenario's configured ×5.5 / ×90. On this older board the *configured
   energy spread is nearly matched* (81 vs 90), while the configured latency
   spread understates reality by ~9×. The same scenario is simultaneously
   optimistic and pessimistic depending on the deployment hardware — a
   cross-board argument that per-board grounding, not one universal constant
   set, is the honest path.
3. **768×576 scales ≈ pixel area on this board too** (×2.0–2.03 latency,
   ×1.94–2.03 marginal energy at 2.25× pixels — slightly sublinear, vs ×2.13–2.25
   on Orin).
4. **Energy per inference is again nearly mode-flat for the big models**
   (dlv3_r101 total: 10.2 J @15W, 9.5 J @30W_ALL, 12.2 J @MAXN) — the Orin
   finding replicates on Volta-generation silicon.
5. **Cold-process costs are far heavier than on Orin**: torch import 6.0 s
   (vs 2.05), CUDA context init 3.9 s (vs 0.10), checkpoint load 0.7–4.1 s,
   **first inference 4.9–7.8 s** (vs 0.35–0.79 s) — a cold local fallback on this
   class of hardware costs ~16–23 s end-to-end, mission-relevant at a 900 s
   deadline. CAVEAT: in these short cold runs the post-first-inference "steady"
   latency matches the sweep for fcn_r50/fcn_r101 but is 2–8× slower for the
   other four models (e.g. lraspp 112 vs 22 ms; dlv3_r101 4.25 s vs 0.54 s) —
   consistent with the GPU governor not ramping clocks in a short-lived process.
   The cold *breakdown* is therefore clock-state-entangled; A7 (`jetson_clocks`
   pinning) is the experiment that would separate autotune from DVFS. Reported
   as observed.
6. **All-6-resident works on Xavier once checkpoints live on the NVMe**:
   CUDA allocated 794 MB / reserved 1.11 GB, process RSS 2.96 GB (heavier
   runtime than Orin's 1.13 GB). Round-robin ≈ dedicated for the big models
   (r101: 534.9 vs 535.1 ms); lraspp shows a small penalty (22.5 vs 19.8 ms).
   Resident switching is effectively free here too.
7. **Client side (same synthetic content — byte-identical payload sizes as Orin,
   as expected from seeded generation; times 2–6× slower):** preprocessing
   36.8/42.1 ms (resize-dominated; >1.5× a LRASPP inference), dense-JSON mask
   decode 100 ms at 768×576 vs RLE 6.4 ms / PNG 2.2 ms, protocol envelope
   ~95 µs. Every Orin conclusion (JPEG ≪ configured 2.0 MB, dense-JSON
   unusable as a wire format, envelope negligible, preprocessing non-negligible)
   holds with more force on the slower CPU.

## Caveats (Xavier)

1. MODE_10W was **not measured** (reboot-gated). MAXN idle floor is higher
   (8.3 W vs 5.9 W) because MAXN holds clocks up; three jtop instances (another
   user's) ran throughout, as on the Orin session-1 measurements.
2. Software stack is a generation older (torch 2.1 vs 2.8, cuDNN differences);
   cross-board ratios mix silicon *and* stack effects — label any comparison
   accordingly.
3. Same synthetic-content / software-codec / no-thermal-soak limits as the Orin
   campaign; nothing here is a UAV flight measurement.
4. The `/images` NVMe hosts Docker state and other users' directories; only
   `/images/aerobench/` (rajat-owned) is ours. Root eMMC remains ~99 % full —
   flagged to the lab, not touched.

---

# Xavier session 2+3 (2026-08-06) — cadence energy, DVFS isolation, TX visibility, thermal soak

Closes the remaining plan items **on the Xavier** (A4, A7, B4, A6; MODE_10W stays
reboot-gated and unmeasured). Scripts: `measure_cadence.py`, `measure_net.py` +
`measure_net_recv.py` (discard receiver, ran on the Orin), `measure_thermal.py`,
plus `measure.py`/`measure_switch.py` re-runs under pinned clocks. Raw:
`results/jetson_power_xavier/raw/` (cadence/net/thermal/rep7) and `raw_jc/`
(pinned-clock cells — separate directory, `*_JC` mode labels; never pool with
default-governor cells). Board left at MODE_30W_ALL with the default governor
(verified: `nvhost_podgov`, GPU idling at 114.75 MHz min freq).

## A4 — energy per mission-second at cadence 1/2/3 (60 s windows, MODE_30W_ALL)

| workload | cadence | inferences | E per mission-second | marginal vs idle |
|---|---|---|---|---|
| none (pure idle window) | — | 0 | 6.39 J/s | — |
| lraspp | 1 s | 60 | 6.44 J/s | ~0.00 J/s* |
| lraspp | 2 s | 30 | 6.51 J/s | ~0.00 J/s* |
| lraspp | 3 s | 20 | 6.50 J/s | ~0.00 J/s* |
| deeplabv3_resnet101 | 1 s | 60 | 11.73 J/s | 5.05 J/s |
| deeplabv3_resnet101 | 2 s | 30 | 8.41 J/s | 1.86 J/s |
| deeplabv3_resnet101 | 3 s | 20 | 7.50 J/s | 0.98 J/s |

\* Below measurement resolution: the 60 s window mean landed *under* the 10 s idle
baseline (background variation ~0.3 W exceeds LRASPP's ~0.16 J/s contribution).

**Findings.** (a) **The idle floor dominates mission energy**: the board burns
~6.4 J every second doing nothing — over a 900 s mission that is ~5.8 kJ the
per-call battery model never charges, and running the light model at 1 Hz is
*indistinguishable from idle* at this instrumentation's resolution. (b)
Duty-cycling only matters for the heavy tier: dropping the strong model from
cadence 1 to 3 saves ~4.2 J/s (11.7 → 7.5), but even at cadence 1 the strong
model only ~1.8× the idle floor. What `budget_planner`-style duty-cycling
actually saves is the *marginal* column, not the total. (c) Marginal per call at
cadence 1 (5.05 J) is ~20 % below the back-to-back sweep marginal (6.33 J) —
race-to-idle between 1 Hz calls recovers a little energy but nowhere near
proportionally.

## A7 — DVFS isolation with `jetson_clocks` (pinned vs default governor)

| cell | pinned (3 reps, min–max) | default-governor sweep |
|---|---|---|
| lraspp @30W_ALL | 20.2–20.4 ms | 22.1 ±0.7 ms |
| fcn_r50 @30W_ALL | 242.0–242.2 ms | 242.0 ±0.1 ms (settled) |
| dlv3_r101 @30W_ALL | 535.9–536.3 ms | 535.9 ±0.1 ms |
| lraspp @MAXN | 11.9–11.9 ms | 13.2–14.2 ms (+112 ms transient rep) |
| dlv3_r101 @MAXN | 367.1–368.2 ms | 370.8 ±0.2 ms |
| **cold** lraspp steady | **20.0 ms** (first inf 4.7 s) | 111.7 ms in the unpinned cold run |
| **cold** dlv3_r101 steady | **536.1 ms** (first inf 4.1 s) | 4,247.8 ms in the unpinned cold run |

**Findings.** (a) **The Xavier cold-run steady anomaly is fully explained by
DVFS clock state**: under pinned clocks the cold-process steady latencies match
the sweep exactly (20.0 vs 22.1; 536.1 vs 535.9), whereas unpinned cold runs sat
at 2–8× — the governor simply does not ramp for a short-lived process. The
**first cold inference remains 4.1–4.7 s even with clocks pinned**, so that part
is genuine one-time work (cuDNN algorithm selection etc.), not clocks. (b) The
MAXN jitter cells tighten completely under pinning (11.9–11.9 ms) — the earlier
variance was governor behaviour, not the silicon. (c) The ≥10 s continuous sweep
cells were long enough to ramp: pinning changes their means by ≤9 %, so the
sweep tables stand as valid steady-state numbers. Post-restore spot check
(rep7, default governor): 20.7 ms / E_marg 0.162 J — consistent with the sweep.

## B4 — transmit-energy visibility: NOT measurable with this instrumentation

TCP stream from the Xavier to a discard receiver on the Orin over the lab LAN:
sustained **942 Mbps (full rate), 50 Mbps, and 10 Mbps phases each show *negative*
mean power deltas (−0.39 to −0.51 W) versus the in-run idle phases** — the NIC/PHY
power either does not flow through the INA3221 module rails or is below the
~0.3 W noise floor. Verdict recorded per the plan's STEP 1 gate: **transmit energy
cannot be grounded on this devkit**; the configured `uplink_energy_j_per_mb`
(60 J/MB, a UAV-radio stress value) stays configured-and-labelled, and the gap
(devkit gigabit Ethernet ≠ UAV LTE/WiFi radio) is now documented with data.

## A6 — thermal soak (900 s back-to-back dlv3_r101, MODE_30W_ALL)

1,687 inferences over 900 s with zero cadence gaps (worst case vs any real duty
cycle): first-minute latency 533.3 ms vs last-minute 533.0 ms — **drift −0.05 %,
i.e. none**; hottest thermal zone peaked at **50.0 °C**, far from throttle
thresholds. At this power mode the devkit's passive thermal mass absorbs a full
mission-length sustained load without measurable degradation. Late-mission
latency drift is a non-issue on this board/mode combination (lab ambient, devkit
heatsink — an enclosed airframe in summer is a different thermal problem, and
MAXN was not soaked).

## Session 2+3 caveats

1. A4 windows are 60 s each, one window per (model, cadence) — no repeats; the
   lraspp marginal is bounded by instrumentation noise (~±0.3 W), not measured.
2. B4's negative deltas mean "invisible on these rails", not "free"; no radio
   claim of any kind follows from it.
3. A6 soaked one model at one mode in lab ambient with the stock heatsink.
4. All JC (pinned-clock) records live in `raw_jc/` with `*_JC` labels; they are
   a *diagnostic* population and must not be mixed into default-governor tables.

---

# Xavier MODE_10W addendum (2026-08-06) — the reboot-gated fourth mode

With two owner-run reboots (into MODE_10W and back to MODE_30W_ALL) the Xavier
grid is complete: **4 modes × 6 models × 3 reps**. The 10W cells (512×384,
same protocol; checkpoints from the NVMe cache):

| mode | model | latency ms (±std) | p95 ms | idle W | run W | E_total J/inf (±) | E_marginal J/inf (±) |
|---|---|---|---|---|---|---|---|
| MODE_10W | lraspp_mobilenet_v3_large | 45.6 ±1.9 | 51.2 | 6.21 | 8.46 | 0.386 ±0.014 | 0.102 ±0.003 |
| MODE_10W | deeplabv3_mobilenet_v3_large | 113.8 ±0.3 | 114.9 | 6.18 | 8.25 | 0.938 ±0.043 | 0.235 ±0.043 |
| MODE_10W | fcn_resnet50 | 781.5 ±0.1 | 781.9 | 6.16 | 8.56 | 6.691 ±0.076 | 1.880 ±0.075 |
| MODE_10W | deeplabv3_resnet50 | 1237.1 ±0.1 | 1239.3 | 6.07 | 8.57 | 10.603 ±0.025 | 3.092 ±0.155 |
| MODE_10W | fcn_resnet101 | 1298.4 ±0.0 | 1298.8 | 5.61 | 8.40 | 10.904 ±0.014 | 3.619 ±0.340 |
| MODE_10W | deeplabv3_resnet101 | 1754.6 ±0.2 | 1757.5 | 5.32 | 8.58 | 15.056 ±0.034 | 5.721 ±0.234 |

**Findings.** (a) At 10 W the strong tier is out of the mission envelope
entirely: dlv3_r101 takes **1.75 s per 512×384 inference** (≈4 s extrapolated to
the zoo's 768×576 by the measured area scaling) at **15.1 J total per call** —
the worst latency AND the worst total energy of any Xavier mode, because the
~6 J/s idle floor accrues for the whole stretched inference. Capping power makes
the big models cost *more* energy per frame, not less. (b) The marginal energy
ranking still favours lower modes (0.102 J lraspp), so the light tier remains
usable at 10 W (45.6 ms). (c) The mode ladder is strongly non-linear for heavy
models: r101 latency 1754.6 → 535.9 → 370.8 ms across 10W → 30W_ALL → MAXN.

**Operational notes for future sessions (both discovered the hard way).**
Every Xavier reboot resets two permissions that must be re-applied before
measuring: (1) INA3221 hwmon rails need `sudo chmod -R a+r`, and (2) **the
`/images` NVMe root comes up `drwx--x---` root-owned** (boot-time state,
reproduced on both reboots) — `sudo chmod a+x /images` is required or the
rajat-owned checkpoint cache at `/images/aerobench` is unreachable and torch
falls back to downloading (or fails with PermissionError, which cost this
session one aborted sweep run). Board verified back at its default MODE_30W_ALL
with the default governor after the second reboot.

# Xavier catalog campaign (2026-08-13) — split heads vs onboard tiers, all eight power modes

The model-catalog workloads (`docs/v2_design.md` §10.11): onboard fp32/fp16
tiers at catalog resolutions, Mask R-CNN onboard, and the onboard **head** of
each published split, measured with the sweep protocol (`measure_catalog.py`;
10 s idle baseline, 5 warmups, ≥30 iters and ≥10 s timed, INA3221 at ~20 Hz).
Split-head timed region = the real head compute — encoder forward + entropy
coding / quantization — and the recorded payload is the bytes the head actually
encoded. Heads are **transcriptions** (`split_heads.py`; `sc2bench` does not
install on the board's py3.8): `verify_heads.py` pinned all four
**byte-identical** to the Mac real backends before any measurement (ES CUDA
symbols 0/289,560 mismatch; compressai 1.2.6 vs 1.2.8 compress byte-identical
with a `_matrix0→matrices.0` rename shim).

Matrix: 9 workloads × 8 modes (every nvpmodel entry, including the 30W
core-count variants and 15W_DESKTOP) × 3 reps = 216 cells; **all 216
measured**. The MODE_10W × maskrcnn cells came last (2026-08-14, one extra
owner-run reboot into 10 W): the original 10W window ran before the torchvision
CUDA build finished. Raw
records: `results/jetson_power_xavier/raw_catalog/` (local-only); aggregate:
`jetson_catalog_summary.json` via `aggregate_catalog.py` (full per-cell table
there — headline rows below).

| mode | dlv3_fp32 768² ms / J | dlv3_fp16 768² ms / J | lraspp_fp16 ms / J | es_b064 head ms / KB | ghnd_bq3 head ms / KB | fcm head ms / KB | maskrcnn ms / J |
|---|---|---|---|---|---|---|---|
| MODE_10W | 2415 / 20.4 | 504 / 4.5 | 39.5 / 0.28 | 69.7 / 21.4 | 27.7 / 39.6 | 237.7 / 4190 | 492 / 4.4 |
| MODE_15W | 977 / 13.8 | 218 / 3.4 | 21.8 / 0.22 | 58.6 / 21.5 | 12.2 / 39.6 | 102.6 / 4190 | 255 / 3.5 |
| MODE_30W_ALL | 734 / 12.9 | 168 / 3.3 | 27.6 / 0.25 | 68.7 / 21.5 | 28.0 / 39.6 | 99.3 / 4190 | 200 / 3.3 |
| MAXN | 506 / 16.5 | 124 / 4.2 | 11.7 / 0.22 | 41.8 / 21.5 | 7.4 / 39.6 | 68.0 / 4190 | 143 / 4.1 |

**Findings.**

- **fp16 is what rescues the strong tier on Xavier.** dlv3_r50 at the catalog's
  768×576 runs ×4.4–4.8 faster in fp16 at every mode (2415→504 ms at 10 W,
  734→168 ms at 30 W) at ~¼ the energy. fp32 cannot sustain the 1 Hz cadence in
  *any* Xavier mode (best 506 ms at MAXN is marginal); fp16 sustains it
  everywhere except 10 W. The catalog's "quantized tier" is not an optimization
  footnote on this board — it is the difference between having and not having a
  strong onboard option.
- **Published split heads are cheap onboard.** The GHND-BQ head costs
  7.4–28 ms / ≤0.26 J total; the ES heads 40–70 ms (entropy coding is CPU-side,
  hence mode-sensitive but never dominant). Both are far below any full strong
  tier while emitting 2.3–39.6 KB payloads. The FCM Mask R-CNN head is the
  outlier in the other direction: moderate compute (68–238 ms) but a fixed
  **4.19 MB/frame** payload — measurement confirms the round-2 finding that the
  standard's split point, without its codec, is priced out by the wire, not the
  head.
- **ES payload bytes are content-dependent; GHND's are not.** On the sweep's
  uniform-random input the ES β0.64 head emits 21.4–21.5 KB vs the 12.6 KB the
  same head produced on mission frames (entropy coding compresses noise worse);
  GHND-BQ (fixed-size 8-bit bottleneck quantization, no entropy stage) emits
  exactly 39.6 KB on both. Catalog payload numbers must say which input they
  came from; the sweep's are labelled random-input.
- **Mask R-CNN onboard is measured at 3.3–4.4 J total per call across the whole
  mode ladder** (143 ms at MAXN to 492 ms at 10 W) — two orders of magnitude
  below the 800 J configured value the catalog scenario currently carries, which
  was a stress placeholder. Notably it degrades far more gracefully at 10 W than
  dlv3_fp32 (×2.5 vs ×3.3 from 30 W, and it stays within a 1 Hz cadence).
  Grounding that config row is now possible and pending the usual owner
  decision.
- **The four 30W core-count variants share GPU clocks, and the GPU-bound cells
  prove it**: dlv3_fp32 is 734.2–734.4 ms across 2CORE/4CORE/6CORE/ALL. Light
  CPU-touching workloads differ only through governor behaviour — MODE_30W_ALL
  (8 cores at lower clocks) shows the familiar DVFS jitter on light loads
  (lraspp_fp32 ±35.6 ms, ghnd head ±31.7 ms across reps; same A7 mechanism,
  governor-entangled, not thermal), while the core-count variants are clean.
  For GPU workloads the 8-mode grid confirms the nvpmodel.conf reading: 4
  GPU-distinct points, the rest CPU-topology variants.
- **Board-side gotchas this campaign added**: NVIDIA torch 2.1 aarch64 **CPU**
  conv produces NaN in whole channels (mkldnn off too) — verify on CUDA only;
  the pip torchvision 0.16.2 wheel has no compiled C++ ops, so Mask R-CNN dies
  at NMS — fixed with a FORCE_CUDA sm_72 source build (0.16.2+c6f3977, in
  `/images/aerobench/build/vision`); INA3221 chmod must cover `in*_label`, not
  just the `*_input` files.

Nothing remains in this campaign. After the final 10W cells the board was
rebooted back and verified at its default MODE_30W_ALL, default governor;
INA3221 rails and `/images` re-opened for the next session (both reset again
on any reboot).

## Xavier resolution-tier addendum (2026-08-21, MODE_30W_ALL)

Four catalog rows for the input-resolution knob (same checkpoints, same protocol;
raw in `raw_catalog/`, aggregated in `jetson_catalog_summary.json`):
dlv3_r50 fp32@512×384 373.0 ms / 4.25 J marginal (cross-checks the original
sweep's 373.2 ms), fp16@512 84.5 ms / 0.96 J; lraspp fp32@768×576 33.9 ms /
0.33 J, fp16@768 28.0 ms / 0.18 J. Consumed by
`make_catalog_profile.py --extended` (docs/v3x_extensions.md §6.3).

# Xavier extended-catalog campaign (2026-08-21→24) — ONNX rows CPU-side, all live modes

The resolution tiers (7 remaining modes) and the six published ONNX rows
(DeepLabV3+-MobileNet / SegFormer-B0 / FFNet-40S, w8a8+float) measured across the
seven live power modes: 198 cells, 3 reps each, resumable driver
(`run_catalog_ext.sh`), raws in `raw_catalog/`, aggregate in
`jetson_catalog_summary.json`. **MODE_10W completed 2026-08-24** with two
owner-run reboots (30 more cells): ONNX rows stretch to 3.4–8.9 s (w8a8 slower
than float here too — the int8 loss now holds across ALL EIGHT modes), and
FFNet-w8a8 reaches 50.7 J TOTAL per frame at 10 W — the power-cap paradox
(idle accruing over a stretched inference) at its extreme. Resolution tiers at
10 W: dlv3 fp32@512 1237 ms / fp16@512 241 ms, lraspp fp32@768 93 ms /
fp16@768 61 ms. ONNX cells run onnxruntime **1.18.1** CPU EP at
ORT_ENABLE_BASIC — the same execution path as the benchmark's
`onnx_semantic_segmentation` kind.

Headline rows (E_marginal J/inf; latency ms):

| workload | 2CORE | 4CORE | 6CORE | 30W_ALL | MAXN |
|---|---|---|---|---|---|
| onnx_dlv3plus w8a8 | 3073 / 5.69 | 1769 / 4.05 | 1651 / 3.18 | 932 / 2.34 | 487 / 4.20 |
| onnx_dlv3plus float | 2655 / 5.06 | 1571 / 3.64 | 1441 / 2.84 | 826 / 2.10 | 438 / 3.81 |
| onnx_segformer w8a8 | 3439 / 5.87 | 2001 / 4.23 | 1848 / 3.36 | 820 / 1.93 | 454 / 3.18 |
| onnx_segformer float | 2172 / 3.77 | 1278 / 2.58 | 1170 / 1.99 | 594 / 1.24 | 331 / 1.92 |
| onnx_ffnet40s w8a8 | 5118 / 9.88 | 3144 / 7.71 | 3090 / 5.93 | 1855 / 4.77 | 991 / 8.80 |
| onnx_ffnet40s float | 4658 / 9.07 | 2887 / 7.15 | 2828 / 5.73 | 1729 / 4.29 | 916 / 8.11 |

Findings:

1. **Published w8a8 is SLOWER than float in every mode, for every model, on this
   board's CPU runtime** (worst: SegFormer, ×1.38 at 30W_ALL). The Mac CPU had
   shown a model-dependent split (FFNet faster, DeepLabV3+ slower); the
   deployment board settles it as a uniform loss — the published quantization's
   latency benefit targets NPUs, and adopting the artifact does not import the
   accelerator it was quantized for. (Direct sibling of the FCM finding: a
   standard's split point does not import its codec.)
2. **The core-count power modes finally matter.** GPU workloads tie across the
   30W variants (734 ms all four); the CPU-bound ONNX rows scale ×3.3 from
   2CORE to 30W_ALL — mode choice and execution stack interact, and a policy
   that could pick power modes would face a genuinely different landscape for
   ONNX rows than for CUDA rows.
3. MAXN trades energy for latency on CPU rows too (fastest everywhere, but
   marginal J/inf roughly doubles vs 30W_ALL — the 10.8 W idle floor and higher
   CPU clocks both bill the same inference).
4. Mission-scale consequence (grounded into `demo_img1_model_catalog_xavier_ext`):
   at 30W_ALL the ONNX rows are 0.59–1.86 s tiers — FFNet now skips every other
   1 Hz slot, and every ONNX row is slower AND costlier than the CUDA
   `local_strong_fp16` (0.167 s / 2.16 J), so their catalog role is honest
   diversity (architecture + published-precision axes), not Pareto competitiveness.

Operational note: the campaign hit two incidents, both recorded — a double
launch (guard added; contaminated cells wiped and re-measured) and onnxruntime
1.19.2 aborting with a C++ vector assertion whenever a power mode takes CPU
cores offline (fixed by pinning 1.18.1; reproduces on 6CORE/4CORE/2CORE/15W).

## Xavier LLM policy decision cost (2026-08-24, MODE_30W_ALL)

First board measurement of what the LLM policy's own decision costs
(`measure_llm_policy.py`; representative mission prompt — grand-tour template,
~900 tokens, 20 option ids; INA3221 protocol as everywhere; fp16 CUDA,
transformers 4.46 on the board stack; results in
`results/jetson_power_xavier/llm_policy_cost/`):

| model | mode | s/decision | E_total | E_marginal |
|---|---|---|---|---|
| Qwen2.5-0.5B | choice (20 fwd) | 8.95 | 177 J | 122.6 J |
| Qwen2.5-0.5B | generate (greedy 24 tok) | 1.08 | 14.5 J | 7.6 J |
| Qwen2.5-1.5B | choice | 17.60 | 325 J | 217.3 J |
| Qwen2.5-1.5B | generate | 3.68 | 51.2 J | 28.5 J |

Readings:

1. **On the deployment board, one 1.5B choice-mode decision costs ~100× the
   perception inference it selects** (217 J vs strong-fp16's 2.16 J), and a
   440-decision grand tour at that rate would burn ≈26.6 Wh on decisions alone
   — more than the mission's whole 21 Wh battery. The Mac diagnostic (3.31 s)
   understated the board latency ×5.3.
2. Generate mode is the only remotely deployable shape (0.5B: 1.08 s / 7.6 J),
   and even it costs ~3.5× the inference it chooses and overruns the 1 s slot.
3. Implementation caveat, recorded: choice mode re-runs the full prompt per
   option (faithful to the current llm_policy backend). Prompt-KV reuse would
   collapse choice toward generate cost — the measured number prices the
   implementation, not the theoretical minimum.
4. The board torch build has no torch.distributed, so transformers'
   ``generate()`` is unusable there; the measurement uses a manual greedy loop
   (same compute). numpy from pip must NOT shadow the system numpy
   (``typeDict`` API removal breaks the NVIDIA stack) — both recorded as
   board gotchas.

# Orin catalog campaign @ MAXN (2026-08-24) — second board, one mode

First catalog-workload data on the AGX Orin: all 19 workloads × 3 reps at the
mode the board was found in (MAXN; Orin mode changes need reboots, deferred).
Setup was fully NVMe-hosted (`/mnt/work/aerobench`) because the shared rootfs
is at 0 bytes free (other users' data — untouched); board stack: py3.10,
torch 2.8 cu126 (preserved in ~/.local), ORT 1.23.2 + compressai 1.2.6 in a
--target dir. Raws: `results/jetson_power/raw_catalog_orin/` — NEVER pooled
with Xavier directories.

Cross-board findings:

1. **The published-w8a8 CPU loss holds on the second board too**: dlv3+ 232 vs
   190 ms, SegFormer 257 vs 127 ms (×2.0!), FFNet 514 vs 441 ms — on modern
   A78AE cores with a current ORT (1.23). Two boards, two ORT generations,
   eight+one modes: int8-on-CPU is a uniform regression for these artifacts.
2. **Orin's CPU changes the ONNX rows' mission viability**: 4–8× faster than
   Xavier (dlv3+ 232 ms vs 932; FFNet 514 vs 1855) — every ONNX row fits the
   1 Hz cadence on Orin at MAXN. Config viability is a board property, again.
3. Split heads are almost free on Orin (GHND 3.3 ms / 0.09 J; FCM 29 ms) and
   reproduce the exact wire payloads byte-for-byte (21.5 / 2.3 / 39.6 /
   4190.2 KB) — the transcription verification holds across boards.
4. Operating-point nuance: lraspp fp16 is SLIGHTLY SLOWER than fp32 on Orin
   MAXN (16.1 vs 14.7 ms) — fp16 does not help models this small there, while
   halving dlv3 (101→54 ms).

Gotchas recorded for future Orin sessions: rootfs 0 bytes → every python
invocation needs TMPDIR on the NVMe; pip --target pulls its own torch (delete
torch*/nvidia*/triton* from the target dir, but keep torch_geometric); pip's
numpy must not shadow the system stack.
