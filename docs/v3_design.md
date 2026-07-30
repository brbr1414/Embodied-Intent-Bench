# V3 design — completing the benchmark

V2 froze with a validated closed loop and a multi-seed result, but with two contract
axes dormant: nothing could spend communication and nothing could violate privacy.
V3 completes the benchmark (the 3D simulator is V4). This document records the V3
milestones; P1 is implemented.

## P1 — network-aware remote inference with deployment-ready boundaries

**Research question.** When remote inference is possible but the link, the budget, and
the contract's privacy level all push back, does configuration selection quality decide
the mission? After P1 all five constraint axes — quality, deadline, battery,
communication, privacy — are operational.

### Architecture (deployment boundary)

Remote execution is a distinct path with explicit stages, never "a local executor with
a bigger constant":

```text
capture -> request encoding -> upload -> network transit -> remote compute
        -> response transit -> response decoding -> completion or failure
```

Three interfaces (`aerointentbench/v2/remote.py`) keep the simulation replaceable by a
real deployment without touching the runner, policies, or evaluator:

- **`InferenceTransport`** — the network. `SimulatedTransport` implements it
  deterministically; a future `HttpTransport`/`GrpcTransport` implements it for real.
- **`RemoteInferenceBackend`** — the server. `SimulatedRemoteBackend` runs an existing
  local model-strategy in-process; a future `RemoteInferenceService` runs on a real
  machine.
- **`SimulatedRemoteExecutor`** — the benchmark-facing executor
  (`run_with_context(rgb, ExecutionContext)`); local executors keep `run(rgb)`.

Future deployment mapping (explicitly **not implemented** in P1 — no server, no RPC
framework, no Jetson):

```text
Camera
  v
Jetson edge runtime
  |-- policy (unchanged: config_id in, config_id out)
  |-- local executors        (local_light / local_strong)
  \-- remote client          (InferenceTransport impl)
          v  real network
Remote inference service     (RemoteInferenceBackend impl)
          v
Result returned to the edge runtime
          v
Benchmark coordinator: evidence, resource accounting, mission evaluation
```

Protocol models are wire-representable (`InferenceRequest.to_wire()` — JSON scalars,
stable request ids, payload by reference + metadata, `protocol_version "1.0"`). The
runner stays synchronous; concurrent in-flight requests would need a `submit/poll`
transport, which the boundary permits later (request ids and capture/completion times
are already explicit).

### Latency semantics (Model A, deterministic)

All stage components come from the **capture-time network snapshot** — a regime change
during a transfer does not affect an in-flight request (documented limitation; an
interval-integrating Model B can replace the transport behind the same interface):

```text
total = request_encoding_s
      + upload_mb x 8 / uplink_mbps
      + rtt_ms / 1000                      # charged exactly once, end to end
      + remote_queue_s + remote_compute_s
      + download_mb x 8 / downlink_mbps
      + response_decoding_s
```

Zero/unavailable bandwidth never divides: it is the `UNREACHABLE` path. Every component
is separately observable in diagnostics and replay (`latency_breakdown_s`). Labels:
mission latency is **derived**; wall-clock stays a separate measured diagnostic and
never drives the mission clock.

### Failure, timeout, and accounting rules (each tested)

| outcome | when | time charged | upload charged | download charged |
|---|---|---|---|---|
| `SUCCESS` | total <= timeout | total | full | full |
| `UNREACHABLE` | link down at request | `unreachable_detect_s` | none | none |
| `UPLOAD_FAILED` | loss > `max_loss_frac` | encode+upload+RTT | **full** (transmitted, lost) | none |
| `REMOTE_ERROR` | backend exception | through remote compute | full | none |
| `TIMEOUT` | total > timeout | exactly `timeout_s` | **proportional to transfer time** | proportional |

A failed remote attempt without a fallback is `success=False` plus a status — never a
silent empty prediction; the runner counts it as a failed inference. No automatic
retries in P1 (documented choice; the policy simply decides again next capture).

### Communication energy

```text
communication_energy_j = uploaded_mb x uplink_energy_j_per_mb
                       + downloaded_mb x downlink_energy_j_per_mb
                       + radio_activation_j        # every attempt, incl. failed probes
```

Configured simulation parameters (never hardware claims); charged on actual simulated
transfer progress; distinct from onboard compute energy and flight energy; drains the
same battery; reported as `communication_energy_j` in results, records, and replay.
The interface later permits Jetson power measurement / radio telemetry to replace it.

### Dynamic network (`aerointentbench/v2/network.py`)

A scenario may declare `simulation.network_trace`: named, piecewise-constant regimes
(`5g_good`, `lte_degraded`, `disconnected`, `recovered`, …) with separate
uplink/downlink, RTT, and loss — conceptually V1's trace model adapted to V2.
Half-open segments; boundary times belong to the regime that starts there. **The
policy sees only the current sample projected into the frozen V1
`NetworkObservation`** (bandwidth = uplink, RTT, loss): no regime names, no
boundaries, no futures, no remote outcomes. Scenarios without a trace keep their
constant network, byte-for-byte unchanged.

### Fallback semantics

The action-level fallback (ActionValidator) is unchanged. The new **executor-level**
fallback: a remote config may name a local `fallback_config_id`; on remote failure the
local model runs **on the same captured frame**; remote elapsed time, transmitted
bytes, and radio energy stay charged; fallback compute time and energy are added; the
UAV kept moving throughout (completion time covers both attempts); the result records
requested config, remote status, and the fallback path — replay shows
`requested: remote_strong -> remote result: timeout -> fallback: local_fast`.
The scenario-level safe fallback must always be local (validated at load).

### Privacy — the fifth hard constraint (V3 semantic activation)

V2 scenarios may now declare `mission_contract.privacy_level`; the frozen V1 logic
(`privacy_permits`) applies verbatim because the catalog gives remote configs
`Placement.REMOTE` and an explicit `transmitted_payload: raw_input`:

| contract \ config | local | simulated_remote (raw RGB) |
|---|---|---|
| `remote_allowed` | allowed | allowed |
| `features_only` | allowed | **forbidden** (raw is not features; fails closed) |
| `local_only` | allowed | **forbidden** |

A forbidden selection is *blocked* (the validator substitutes; the transfer is never
simulated), *counted* as an invalid action with `privacy_violation` outcome, and the
mission's privacy constraint fails on the record — exactly V1. Mission success is now
the five-way conjunction:

```text
mission_success = quality AND deadline AND battery AND communication AND privacy
```

Compatibility: local-only scenarios gain the fifth key trivially satisfied — every
previously successful V2 mission stays successful (pinned by tests). Replay reports
privacy `NOT_APPLICABLE` only for scenarios with no remote path; otherwise
SAFE/VIOLATED from the mission record. `features_only` remote execution with a genuine
feature payload is **not implemented** (no feature extractor exists; the honest
mapping is "raw upload forbidden").

### Validation scenarios (all deterministic, CI-safe on the tiny test world)

Covered by `tests/test_v2_remote_network.py` (27 tests) and the committed demo
`data/v2_scenarios/demo_img1_remote_network.json` (heuristic executors — no Torch):

| case | behaviour demonstrated |
|---|---|
| A remote advantageous | good link: remote succeeds, comm charged, privacy passes |
| B degradation | `5g_good -> lte_degraded -> disconnected -> recovered`: timeouts, unreachable, fallbacks, multiple transitions |
| C budget pressure | `always_remote_strong` breaches the 30 MB budget (34.3 MB) and fails |
| D privacy local_only | remote selection blocked + counted; mission privacy FAIL |
| E timeout + fallback | partial upload charged; local fallback on the same capture; both attempts on the clock |
| F trace boundary | half-open regime boundaries follow Model A exactly |

Demo outcomes (local run, `results/v3_remote/`): `always_fast` succeeds (recall metric
tolerates its 861 false positives); `local_strong` succeeds; `always_remote_strong`
**fails communication** (34.3 MB > 30 MB, 10 remote failures); `rule_based` succeeds —
13 remote calls on the good link, 10 `local_strong` after degradation, **zero failed
remote attempts**: the V1 reachability/affordability/latency-budget rules, dormant
since V2.0, are now exercised.

### What is simulated / configured / derived / measured

- **configured**: stage times, energies (J/MB, activation, codec), timeout, loss
  ceiling, regime parameters, backend compute time.
- **derived**: end-to-end remote latency, transferred MB, communication energy — from
  configured parameters and the network state; deterministic.
- **measured**: wall-clock only, diagnostic only.
- **not validated by P1**: real wireless performance, real server latency, Jetson
  power, RPC overhead. Nothing here is a hardware claim.

### P1 limitations

Model A in-flight invariance; no retries; no concurrent requests; raw-RGB payloads
only; single remote backend per config; remote compute energy not tracked (server-side
cost is not the vehicle's); regime traces are hand-authored, not measured.

## P2 — statistical hardening: remote-aware hard family, two worlds, 100 seeds

**Research question.** Is "adaptive configuration selection decides the mission" a
pattern that survives when all five constraint axes are live, when the world imagery
changes, and when the sample is large enough for meaningful intervals — or was V2.4's
result an artefact of one world and a dormant network?

### Remote-aware hard base scenarios

Two committed bases carry the V2.4 hard trade-off structure **plus** the P1
remote/network path, so quality, deadline, battery, communication, and privacy can all
independently fail:

- `data/v2_scenarios/demo_img1_remote_hard.json` (`V2_IMG1_REMOTE_HARD`) — the V2.4
  world and trajectory.
- `data/v2_scenarios/demo_img2_remote_hard.json` (`V2_IMG2_REMOTE_HARD`) — a second
  OpenAerialMap world (`img_2`, 0.039 m/px, open scrubland field), same trajectory
  geometry, executors, network trace, and contract, so cross-world comparisons vary
  only the background imagery. Late walking targets anchor at 1.9 m (vs 1.5 m on
  img_1) because the light model needs more pixels against this background —
  "comfortably detectable" is a world property, measured before freezing the base.

Design (both bases): 4 small early targets (~32 px, Stage-B band) defeat the light
model; 4 large late targets sit on odd capture slots ≥ 41 s so the strong executor's
2-capture cadence skips them; the link starts good (`5g_good`, remote viable at
~0.6 s derived latency) and degrades (`lte_degraded` at 12 s → `disconnected` →
`recovered_weak`), so remote inference is only ever useful early; the 26 MB
communication budget is sized so an always-remote policy breaches it while the
adaptive policy's affordability rule stops in time; strong compute (600 J/call) and
the radio (60 J/MB uplink, stress-configured like V2.4's accelerator energy) make
battery a live constraint on every path. The intended base pattern, verified with
real models before freezing:

| policy | outcome | failing axis |
|---|---|---|
| `always_light_real` | FAIL | quality (small targets invisible) |
| `always_strong_real` | FAIL | battery (completes, far below floor) |
| `always_remote_strong` | FAIL | communication (~33 MB > 26 MB) |
| `rule_based` | SUCCESS | — via remote → local_strong → local_light |

The adaptive success now takes **two switches** (budget/degradation pressure, then
battery pressure), retiring V2.4's "every success is a single switch" fragility flag
by design rather than by relaxing the flag.

### Family 2.0

`scenario_family.FAMILY_VERSION = "2.0"` generalises the family across bases and adds
network dimensions (full history in the module docstring):

- battery capacity: relative band (base × 0.97–1.03) instead of a hardcoded absolute
  band;
- late-target heights: relative to the base late target of the same pose
  (× 0.93–1.07);
- network (bases with a `network_trace`): regime boundaries jitter ± 2 s, within-regime
  uplink/downlink scale together × 0.75–1.3, RTT × 0.85–1.25. Regime structure, order,
  packet loss, and reachability classes are base identity and never resampled.

All sampled values are recorded under `provenance.scenario_family` (including the full
sampled trace). Family 1.1 variant identities change with the version bump; the V2.4
30-seed result under 1.1 stands as recorded history.

### Evaluation harness

`multi_seed_eval` gains the remote axis: bases with a `simulated_remote` executor
default to the 4-policy set (`always_light_real`, `always_strong_real`,
`always_remote_strong`, `rule_based`); per-run records restate
`communication_mb` / `communication_margin_mb` / `communication_energy_j` /
`network_behaviour` from the mission result; aggregates add communication and privacy
failure rates; the paired comparison generalises to any number of static baselines;
the fragility report adds the adaptive switch-count distribution.

### Results (100 seeds x 4 policies x 2 worlds, family 2.0)

Seeds 0-99, paired by construction, real torchvision models, deterministic configured
latency/energy. Full artifacts in `results/v3_remote_multiseed/` (local-only,
gitignored — frames composite non-redistributable assets).

| world | policy | success | Wilson 95% CI | quality fail | battery fail | comm fail |
|---|---|---|---|---|---|---|
| img_1 | always_light_real | 0/100 | [0.000, 0.037] | 1.00 | 0.00 | 0.00 |
| img_1 | always_strong_real | 0/100 | [0.000, 0.037] | 0.90 | 1.00 | 0.00 |
| img_1 | always_remote_strong | 0/100 | [0.000, 0.037] | 1.00 | 0.00 | 1.00 |
| img_1 | **rule_based** | **55/100** | **[0.452, 0.644]** | 0.30 | 0.33 | 0.00 |
| img_2 | always_light_real | 0/100 | [0.000, 0.037] | 1.00 | 0.00 | 0.00 |
| img_2 | always_strong_real | 0/100 | [0.000, 0.037] | 0.97 | 1.00 | 0.00 |
| img_2 | always_remote_strong | 0/100 | [0.000, 0.037] | 1.00 | 0.00 | 1.00 |
| img_2 | **rule_based** | **32/100** | **[0.237, 0.417]** | 0.56 | 0.40 | 0.00 |

- **Paired patterns**: every adaptive success is an only-adaptive success (img_1: 55
  only-adaptive + 45 all-fail; img_2: 32 + 68). **Zero counterexamples** on either
  world — no seed where any static policy beats rule_based.
- **Class separation is the mechanism, on both worlds** (small / late detection):
  light 0.04/0.78 vs 0.00/0.70, strong 0.99/0.16 vs 0.89/0.11, rule 0.87/0.57 vs
  0.78/0.56.
- **Network awareness**: rule_based averaged 11.7 remote attempts with **zero
  failures** on both worlds (remote only while the link is good); always_remote_strong
  averaged 29.5 attempts with 16.9 failures and breached the budget on every seed.
- **The one-switch flag is retired**: every adaptive success now uses exactly **two**
  switches (remote → local_strong on budget/degradation pressure, → local_light on
  battery pressure). Honest note: the switch count is uniform — the family exercises a
  two-stage escalation, not free-form adaptation; richer behaviour needs the P3
  policy work, not more seeds.
- **Fragility (kept, by design)**: 26/55 (img_1) and 16/32 (img_2) successes clear the
  battery floor by < 0.01. The family deliberately probes the knife-edge; adaptive
  failures (45 and 68 seeds) are preserved in `runs.jsonl` and counted against the
  claim, not excluded.
- **Cross-world sensitivity**: img_2 is markedly harder for quality (0.56 vs 0.30
  adaptive quality-failure rate) — background texture changes model behaviour even
  with identical mission design. This is exactly the evidence the second world exists
  to produce: the qualitative claim (adaptive-only success, statics at 0/100) holds,
  while absolute rates are world-dependent and must never be quoted as a single
  number.

Claim vocabulary (docs/v2_design.md §11): "adaptive beats every static baseline under
live network/privacy/communication axes" is now **supported-across-seeds on two
worlds**; anything about real radios, real servers, or real aerial-human perception
remains **not-evaluated**.

### Honesty

Everything from P1's labels carries over: configured/simulated energies and stage
times, derived latencies, synthetic generated humans, integration diagnostics — never
real UAV, radio, model-serving, or aerial-human-perception performance. The battery
knife-edge is deliberate (the family probes it); seeds where the adaptive policy fails
are preserved, listed, and counted against the claim.

## Later V3 milestones (not implemented)

P3 policy skyline (GT-aware offline upper bound + budget planner);
P4 real-data grounding through the V1 empirical bundle pipeline;
reference deployment (real Jetson client + inference service behind these interfaces).
