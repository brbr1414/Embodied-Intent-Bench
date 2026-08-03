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
