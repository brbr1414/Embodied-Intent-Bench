# Server-side latency profile — development-Mac stand-in (2026-08-17)

**This machine is an Apple-Silicon laptop (fp32, MPS, torch 2.13), a STAND-IN for a
server.** The table's role is to give the coming server model a measured shape
(relative costs, batch scaling, capacity); none of these numbers is production server
performance, and server energy is out of scope by design (not measurable here, and no
UAV constraint reads it). Raw records: `results/server_profile/mac_server_profile.json`
(local-only). Protocol: 5 warmups, ≥30 iters and ≥5 s per cell, seeded random input,
`measure_mac_server.py`.

## Full models (the raw-remote server side)

| model @ input | b1 | b2 | b4 | b8 | per-image @ largest batch |
|---|---|---|---|---|---|
| deeplabv3_resnet50 @768×576 | 170.9 ms | 333.2 | 673.3 | 1374.2 | 171.8 ms |
| lraspp_mobilenet_v3_large @512×384 | 8.0 ms | 13.2 | 26.0 | 53.1 | 6.6 ms |
| maskrcnn_resnet50_fpn @512×384 | 75.5 ms | — | 298.2 | — | 74.6 ms |

## Split tails (the presplit kinds' server half, `complete()` on the real backends)

| tail | latency | payload in |
|---|---|---|
| Entropic Student β0.64 (decode + decoder + layer2..4 + head) | 102.9 ms | 21.4 KB |
| Entropic Student β5.12 | 102.3 ms | 2.3 KB |
| GHND-BQ 3ch | 77.2 ms | 39.6 KB |
| FCM Mask R-CNN (RPN + ROI heads) | 48.0 ms | 4190 KB |

## What shapes the server model

1. **No batch economy for the heavy models on this device.** dlv3_r50 per-image cost
   is flat across batch 1→8 (170.9 → 171.8 ms); Mask R-CNN likewise (75.5 → 74.6).
   Only LRASPP gains (~1.2×). The GPU is saturated at batch 1, so on this stand-in
   the contention model degenerates to pure queueing: capacity = 1/T_inference,
   T_queue ≈ N_waiting × T_inference. A server GPU with real batch economy would
   change that curve — which is exactly why the capacity parameter must come from a
   per-device table, not a formula.
2. **Split tails are cheap but not free** — 48–103 ms, comparable to a full Mask
   R-CNN pass. A server model that only prices full models would flatter the split
   rows.
3. **ES entropy decode costs ~26 ms and is payload-size-insensitive** (β0.64's
   21.4 KB and β5.12's 2.3 KB decode in the same 103 ms; the symbol count, not the
   byte count, drives it). GHND's no-entropy tail saves exactly that stage.
4. **The FCM tail is the cheapest server half (48 ms) behind the most expensive wire
   (4.19 MB)** — server measurement independently reconfirms that the FCM row's
   problem is the link, never the compute.
5. Against the scenarios' configured `remote_compute_s`: raw remote 0.35 s configured
   vs 0.171 measured here; presplit 0.2 s vs 0.077–0.103; FCM 0.3 s vs 0.048. The
   configured values are conservative on this stand-in; per-server grounding will
   replace them when the server model lands.

Open (deliberately deferred): the ServerModel abstraction itself (profiled +
analytical backends, `server_load_trace`, policy-visible advertised estimate) — this
table is its calibration input.
