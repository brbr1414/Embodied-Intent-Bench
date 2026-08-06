# Jetson measurement plan — what is worth measuring, and why

Consolidated list of on-device measurements that would ground quantities the
benchmark currently treats as configured, assumed, or free. Each item names the
benchmark quantity it grounds. Status: only item 0 is done; everything else is
proposed. Nothing in `data/` or `docs/` changes until a measurement lands AND the
owner approves the corresponding value update.

Device: Jetson AGX Orin 64GB (`rajat@128.195.55.171`), back in its default
MODE_30W (found rebooted by another user on 2026-08-05; no restore action remains).
Shared harness: `measure.py` protocol (idle baseline → warmup → ≥30 iters / ≥10 s,
INA3221 rails @ ~20 Hz), extended per item.

**Status: session 1 is DONE (2026-08-05, at MODE_30W — not the originally planned
15W): A1, A2, A3, A5, B1, B2, B3. Results in `RESULTS.md` "Session 1". Remaining:
A4, A6, A7, B4 (session 2/3).**

**Second board (2026-08-05): the full campaign (sweep + session-1 items) was
REDONE on a Jetson AGX Xavier 32GB** (`rajat@128.195.55.180` pw `rajat`, L4T
R35.5.0, NVIDIA torch 2.1 + pip torchvision 0.16.2) — see `RESULTS.md` "Second
board". Xavier deltas that change this plan's assumptions: nvpmodel switches
LIVE for 15W/30W_ALL/MAXN (no reboot — in-mission mode adaptation is possible
there), but MODE_10W is reboot-gated (unmeasured); checkpoints + any large
artifacts belong on the 422 GB NVMe at `/images/aerobench/` (root eMMC is ~99 %
full); the A7 (jetson_clocks) item is now MORE important — Xavier cold-process
runs showed 2–8× clock-state-dependent steady latencies.

**2026-08-06: A4, A6, A7, B4 are DONE ON THE XAVIER** (RESULTS.md "Xavier
session 2+3"): A4 — idle floor ~6.4 J/s dominates mission energy, lraspp@1 Hz is
indistinguishable from idle, strong-tier duty-cycling saves ~4.2 J/s; A7 —
pinned clocks fully explain the cold-run steady anomaly and the MAXN jitter
(sweep tables stand, ≤9 % governor effect); B4 — TX power is NOT visible on the
INA3221 rails even at 942 Mbps (60 J/MB stays configured-and-labelled); A6 —
900 s sustained strong-model soak shows −0.05 % drift, max 50 °C. **On the ORIN
these four items remain unmeasured** (and a 15W 768×576 sweep on the Orin stays
reboot-gated).

**2026-08-06 later: Xavier MODE_10W measured too** (two owner-run reboots; grid
now 4×6×3 complete — RESULTS.md "MODE_10W addendum"). At 10 W the strong tier
leaves the mission envelope (r101 1.75 s / 15.1 J per 512×384 call — power
capping makes heavy models cost MORE energy per frame). **Nothing measurable
remains on the Xavier.** Reboot re-setup for any future session: re-apply
`chmod a+r` on the INA3221 hwmon rails AND `chmod a+x /images` (both reset at
every boot).

## 0. DONE — power-mode × model sweep

4 modes × 6 models × 3 repeats at 512×384. See `RESULTS.md`. Byproduct finding:
**power-mode switching requires a reboot** — adaptive power-mode policies are
impossible on this hardware; the mode is a pre-mission constant.

## A. Model execution (local path)

| # | Measurement | Grounds | Effort |
|---|---|---|---|
| A1 | **DONE (30W)** — 768×576 sweep (fcn_r50, dlv3_r50, dlv3_r101 × 3 reps) | Cost scales ≈ pixel area (×2.13–2.25); zoo-resolution spread 14.4× latency / 38× marginal energy. 15W cells NOT measured — rerun at 15W only if the ratio-grounding decision insists on 15W ratios | see RESULTS.md |
| A2 | **DONE (30W)** — model-switch cost | Non-resident switch = 0.3–2.2 s load + 350–790 ms first inference; all six resident = ~1 GB and swap-free — "switching is free" holds only if the zoo stays resident | see RESULTS.md |
| A3 | **DONE (30W)** — cold-start first inference | 350–790 ms, 3–43× steady state | see RESULTS.md |
| A4 | **Energy per mission-second at cadence 1/2/3** (60 s loops, total J) | The battery model charges per-call J only; measured idle floor (~6.3 W) suggests per-call compute is ~5 % of real system draw at 1 Hz. Quantifies what duty-cycling (budget_planner) actually saves — race-to-idle says little | ~20 min |
| A5 | **DONE (30W)** — preprocessing cost | 12–16 ms, resize-dominated; ~65 % of a LRASPP inference | see RESULTS.md |
| A6 | **Thermal drift**: 15–20 min sustained heavy-model runs at 15W/30W, latency trend + thermal zones | Missions are 900 s; the 10 s cells never reach thermal soak. Answers "does late-mission latency break the contract" | ~40 min wall, mostly waiting |
| A7 | (low) **DVFS jitter**: repeat key cells with `jetson_clocks` pinned | Isolates the governor effect behind the 50W-slower-than-30W anomaly; p95/deadline realism | ~15 min, needs sudo |

## B. Remote path, client side (the `Size_up×J_up + Size_down×J_down + J_act` model and Model A latency)

| # | Measurement | Grounds | Effort |
|---|---|---|---|
| B1 | **DONE** — t_encode + real Size_up (software codecs, synthetic content) | Structured JPEG q85 = 0.028/0.054 MB vs configured 2.0 MB (~40–70× overstated even against the noise upper bound ~6×); t_encode 1.6–6.3 ms. JPEG-vs-raw remains entangled with the privacy premise — still an explicit pending decision | see RESULTS.md |
| B2 | **DONE** — t_decode + Size_down mask formats | Dense JSON 0.59/1.33 MB & 25–57 ms decode vs RLE/PNG ≤4 KB & ≤2 ms (180–3,700× smaller) — wire-format evidence for the future real server | see RESULTS.md |
| B3 | **DONE** — protocol 1.0 (de)serialization | ~350 B, ~15 µs round-trip — negligible | see RESULTS.md |
| B4 | **Transmit energy J/MB**: iperf3 at fixed rates, rail-power delta vs idle. STEP 1: verify network I/O is even visible on the three INA3221 rails; if not, record "not measurable with this instrumentation" | `uplink_energy_j_per_mb` (configured 60 J/MB stress value). HARD LABEL: devkit WiFi/Ethernet is NOT a UAV radio (LTE ~1–10 J/MB, WiFi ~0.01–0.1 J/MB); result documents the gap, never silently replaces the configured value | ~15 min + visibility check |
| B5 | **J_act (radio activation)** | Cellular-modem phenomenon; no modem on the devkit → stays configured, labelled. Listed so the decision is on record | not measurable here |

## C. Explicitly NOT Jetson's to measure

- **t_server_compute** — measured on the server when the real remote backend lands.
- **Real radio RTT / link quality** — no modem; the named network regimes stay
  simulated, which is the honest state.
- **TensorRT fp16/int8 path** — engineering work, not measurement; would change
  the model-zoo ordering itself (likely 3–10× faster). Separate milestone if ever.

## Suggested execution bundles

1. ~~**Session 1:** A1 + B1 + B2 + B3 + A5 + A2/A3~~ **DONE 2026-08-05 at
   MODE_30W** (the board had been rebooted into its default mode by another
   user; the owner chose 30W over forcing another reboot). Scripts:
   `measure_client.py`, `measure_switch.py`, `measure.py` + resolution arg.
2. **Session 2 (~30 min, any mode):** A4 + B4 visibility check.
3. **Session 3 (~45 min, mostly idle):** A6 thermal soak; A7 if sudo convenient.

~~After all sessions: restore the board to its default MODE_30W.~~ Already back
at MODE_30W — no restore action remains. A 15W follow-up (reboot, owner-run) is
needed only if the zoo ratio-grounding decision insists on 15W ratios.
