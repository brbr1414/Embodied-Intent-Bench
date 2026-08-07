# V3 extensions (post-freeze): model zoo and richer policy baselines

V3 froze with P1–P4 (`docs/v3_design.md`). This document records post-freeze
extensions that widen the benchmark's *content* — more selectable models, more
policy baselines — without touching the frozen semantics: the runner, evaluator,
schemas, and V3 results are unchanged.

## 1. Model zoo — six real checkpoints, seven configurations

The `torch_semantic_segmentation` executor resolves any torchvision segmentation
model id (the backend was already generic; no code change was needed). All six
available checkpoints were verified locally (official DEFAULT weights, person class
from weight metadata, forward times measured at 512x384 on Apple silicon, 2026-08):

| model | params | measured fwd | zoo config | configured latency / energy | tier |
|---|---|---|---|---|---|
| lraspp_mobilenet_v3_large | 3.2M | 8.4 ms | local_light_real | 0.4 s / 10 J | low |
| deeplabv3_mobilenet_v3_large | 11.0M | 12.1 ms | local_mid_real | 0.6 s / 60 J | medium |
| fcn_resnet50 | 35.3M | 58.2 ms | local_fcn50_real | 1.1 s / 350 J | medium |
| deeplabv3_resnet50 | 42.0M | 78.1 ms | local_strong_real | 1.4 s / 600 J | high |
| fcn_resnet101 | 54.3M | 96.6 ms | (available, not in demo) | — | — |
| deeplabv3_resnet101 | 61.0M | 116.7 ms | local_heavy_real | 2.2 s / 900 J | high |

Plus two remote tiers: `remote_strong` (DeepLabV3-R50 backend, 2 MB upload) and
`remote_light` (DeepLabV3-MNV3 backend, 0.6 MB upload, 0.1 s server compute).

Configured latencies/energies preserve the **order** of the measured forwards but are
scaled to the 1 s decision interval and stress-scaled for the short mission, exactly
as in the V2.4/V3 hard scenarios — configured simulation parameters, never hardware
claims. Scenario: `data/v2_scenarios/demo_img1_model_zoo.json` (remote-hard mission,
seven-config catalog).

**Headline result** (single scenario, real models): the 61M-parameter
DeepLabV3-R101 scores recall **0.125** — worse than the 3.2M LRASPP's 0.5 — because
its 2.2 s configured latency skips two of every three frames and its energy drains
the battery to 0.07. Bigger is not better under a mission contract; this is the
benchmark's thesis in one row.

| policy on the zoo | outcome | recall | note |
|---|---|---|---|
| local_light_real | FAIL quality | 0.500 | misses all small targets' late window |
| local_strong_real | FAIL quality+battery | 0.500 | cadence 2 + 600 J/call |
| local_heavy_real | FAIL quality+battery | 0.125 | cadence 3: the anti-headline |
| remote_strong | FAIL quality+comm | 0.250 | 33.2 MB > 26 MB |
| rule_based | **SUCCESS** | 0.750 | remote x12 → strong x16 → light x4 |
| budget_planner | FAIL quality | 0.625 | flaps across 26 segments (incl. remote_light) |
| utility | FAIL quality | 0.375 | leaves remote too early; ends on mid tier |
| sticky_escalation | FAIL quality | 0.625 | rule-shaped switches, 1-step timing miss |

## 2. Two new policy baselines

Both are V1-interface policies (contract + frozen `RuntimeState` + public profiles
only), registered as `utility` and `sticky_escalation`; CI-safe tests in
`tests/test_scored_policies.py`.

- **`utility`** (`aerointentbench/policies/utility.py`) — replaces `rule_based`'s
  lexicographic elimination with a scalar cost-benefit score per configuration
  (tier value minus weighted frame-loss, budget-share, and battery-pressure x
  latency costs), argmax per step. Privacy-forbidden and dead-link-remote options
  are excluded, not scored.
- **`sticky_escalation`** (`aerointentbench/policies/sticky_escalation.py`) — the
  same utility score plus **switching hysteresis**: a challenger must out-score the
  incumbent for `dwell_steps` (default 3) consecutive decisions before a switch;
  battery emergencies and a dead-link remote incumbent bypass the dwell. This is the
  direct response to the P3 finding that per-step argmax policies flap and halve the
  strong model's effective cadence.

## 3. Results on the remote-hard family (seeds 0–9, both worlds)

| policy | img_1 | img_2 | note |
|---|---|---|---|
| rule_based (P2/P3 reference) | 5/10 | 4/10 | |
| budget_planner (P3) | 1/10 | 2/10 | flaps |
| **sticky_escalation** | 1/10 | 2/10 | **no flapping** (2 switches); wins are a subset of rule_based's |
| **utility** | 0/10 | 0/10 | per-step argmax leaves remote early, settles on mid tiers |

Honest reading, consistent with P3's "sophistication does not automatically win":

- Hysteresis **fixes the mechanism** (flapping is gone; switch counts match
  rule_based's two-stage escalation) but not the outcome — a one-step difference in
  switch timing loses the marginal late target on most seeds.
- The hard family's late-window structure was co-designed with a contiguous
  escalation of exactly `rule_based`'s timing, so policies with slightly different
  timing are at a structural disadvantage there. That is a property of this family,
  not of the policies — and re-tuning either the family or the new policies' weights
  to flip these rows is exactly what the benchmark's rules forbid.
- What the new baselines add is **spectrum**, not a new champion: static → reactive
  rules → per-step scoring → scored commitment now span four qualitatively different
  adaptation strategies, all separated by the benchmark's metrics (flap counts,
  switch distributions, failure axes), with the GT-aware skyline (recall 1.0)
  bounding them all from above.

Artifacts: `results/v3x_zoo/` (local-only). Full-100-seed runs for the new policies
are a follow-up if a claim beyond "development-grade sample" is ever needed.

## 4. Hardware-grounded profiles (2026-08-06): measured Jetson values as scenario variants

The zoo scenario's configured latency/energy values were deliberate mission-scale
stress settings. After the Jetson measurement campaign
(`experiments/jetson_power/RESULTS.md`: AGX Orin 64GB and AGX Xavier 32GB, both at
their default power modes), the owner-approved grounding decision replaced
assumptions with measurements — as **new scenario variants**, one per board, leaving
`demo_img1_model_zoo.json` and its documented results untouched:

- `data/v2_scenarios/demo_img1_model_zoo_orin.json`
- `data/v2_scenarios/demo_img1_model_zoo_xavier.json`
- generator (rules + sources recorded in provenance):
  `experiments/jetson_power/make_hw_profiles.py`

Derivation: local configs carry **measured absolute values** at each config's own
input resolution (mean latency; **marginal** energy per call — compute only); the
measured idle floor (~6–7 W) is folded into `flight_power_w`, which is where the A4
cadence finding says an always-on board belongs; remote uploads keep the raw-RGB
privacy premise but carry measured raw frame sizes (0.590 / 1.327 MB vs the old
2.0 / 0.6); `uplink_energy_j_per_mb` and all server-side terms stay configured
(B4: TX power is not measurable on the devkit rails). `battery_capacity_wh` (3.0)
is a designed value, labelled as such.

### Verified outcomes (real models, single runs, frozen before documenting)

| policy | Orin profile | Xavier profile |
|---|---|---|
| local_light_real | FAIL quality (4/8, 56 FPs) | FAIL quality (4/8) |
| local_strong_real | **SUCCESS** (8/8) | **SUCCESS** (8/8) |
| local_heavy_real | **SUCCESS** (8/8, 0 skips) | FAIL quality (4/8 — **24 of 48 slots skipped by the measured 1.086 s latency**) |
| remote_strong | FAIL quality (30 obs; raw uploads cap coverage) | FAIL quality (same) |
| rule_based | SUCCESS (= static strong; stays local) | SUCCESS (12 remote → 36 local_strong) |

Honest findings, in order of importance:

1. **The hardware profile decides whether the headline trade-off exists at all.**
   On the Orin at 30 W the model-size dilemma *evaporates* — the 61M-parameter
   model measures 0.271 s / 4.15 J and simply succeeds. On the Xavier the same
   model measures 1.086 s, physically cannot hold the 1 Hz cadence, skips half the
   mission's observations and fails quality — the original zoo's "bigger is not
   better" mechanism reproduced from measurement rather than configuration. A user
   choosing a hardware profile is choosing which regime they are benchmarking.
2. **A well-chosen static suffices on both profiles**: `rule_based` succeeds but
   does not beat `local_strong_real` (on Orin it selects it verbatim). These
   single-scenario diagnostics do not require adaptation — the discriminating
   pressure that motivates adaptive policies lives in the harder families, and no
   knob was turned here to manufacture an adaptive win.
3. **The battery constraint never bound** (worst final fraction 0.28 vs floor
   0.22). Separating heavy from strong on the Orin by battery would need a
   ~75 J (1.2 %) knife-edge — measurement-noise theatre, deliberately not done.
   The binding axes are quality (real model + real latency) and communication
   (raw 1.327 MB uploads cap remote coverage at ~19 calls of the 26 MB budget).

Artifacts: `results/v3x_hw_profiles/` (local-only). Caveats: bench measurements of
inference on devkits (no flight load, no radio, lab thermals); board-default power
modes only; single verification runs, not seed campaigns.

**Telemetry (2026-08-06 addition).** Both profiles enable the periodic
ground-station uplink (`docs/v2_design.md` §10.7): 1 Hz status reports sized on the
measured 353 B protocol envelope (0.001 MB with headroom), transmit energy at the
configured 60 J/MB stress value. Re-verified with telemetry on: every
success/failure outcome above is unchanged (per mission: ~47 report attempts,
exactly 12 lost during the 26–38 s disconnected regime, ~0.035 MB sent — status
traffic is deliberately cheap here; the schema supports heavier reports for
scenarios that want telemetry to genuinely contend for the budget).

## 5. Deployment matrix (2026-08-06): onboard fp32 / onboard fp16 / full server / split

Owner-requested diversification of the model options: every model family can now be
deployed four ways. No new machinery was needed for three of them (onboard fp32 and
the raw-RGB server path existed; onboard fp16 is the torch kind's existing
``dtype`` knob); the new ``simulated_split`` executor kind (`docs/v2_design.md`
§10.8) adds the fourth. Demo:
`data/v2_scenarios/demo_img1_deployment_matrix.json` — the zoo mission under a
**features_only** contract, seven configs: lraspp/dlv3_r50 × {fp32, fp16},
raw-RGB remote (deliberately privacy-forbidden here), and two real split configs
(dlv3_r50 cut at ``layer2`` = 3.539 MB uint8 features; lraspp cut at ``10`` =
0.184 MB including the graph-cut low tap — both exact computed sizes).

Verified outcomes (real models; fp16 and split-head costs are labelled
placeholders pending the lab's measurements — the SAM split-point curves measured
on the same Xavier, and fp16 board sweeps):

| policy | outcome | failing axis |
|---|---|---|
| light fp32 / fp16 | FAIL | quality (0.5) |
| strong fp32 | FAIL | battery AND quality (1.4 s latency skips half the slots) |
| strong fp16 | FAIL | battery only (recall 1.0 — 1.0 s holds cadence!) |
| raw remote | FAIL | privacy (48 blocked selections) + quality |
| split strong / light | FAIL | quality — 3.5 MB features time out off the good regime |
| rule_based | FAIL | battery (recall 0.875; nearest miss) |

Honest findings kept: (a) **no tested policy satisfies this contract** — and that
is documented rather than tuned away, because the attempt exposed something
better: (b) **the battery floor is a moving target for contract-aware policies**
(lowering it from 0.22→0.18→0.17 made ``rule_based`` escalate more aggressively
each time and land just under the floor again — floor placement cannot
manufacture an adaptive win, which is exactly the co-design trap the benchmark's
rules warn about); (c) the matrix separates **all five failure axes across
deployment options**: quality (light, split-over-degraded-link), battery
(strong), latency-via-cadence (fp32 strong skipping), privacy (raw remote), and
the fp16 tier's distinct value (only config with recall 1.0 within cadence);
(d) split's fragility is real dynamics, not configuration — 3.5 MB feature
uploads outlive the 3 s timeout on every regime after the first 12 s, so each
attempt burns head energy plus timeout plus fallback and coverage collapses.
Satisfiability of the contract by some schedule is UNVERIFIED (a skyline run on
this scenario is the open follow-up). Artifacts:
`results/v3x_deployment_matrix/` (local-only).
