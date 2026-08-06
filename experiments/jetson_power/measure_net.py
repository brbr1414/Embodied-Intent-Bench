#!/usr/bin/env python3
"""Transmit-energy visibility check and J/MB estimate (plan item B4, STEP 1 + 2).

Streams data over TCP to a discard receiver at several target rates while sampling
the INA3221 rails, then compares each phase's mean power against the in-run idle
phase. STEP 1 is the honest gate: if the transmit phases are not clearly above the
idle floor, the result is "not measurable with this instrumentation", not a number.

HARD LABEL (from the measurement plan): a devkit Ethernet/WiFi NIC is NOT a UAV
radio (LTE ~1-10 J/MB, WiFi ~0.01-0.1 J/MB); whatever comes out documents the gap
to the configured 60 J/MB stress value and never silently replaces it.

Usage: measure_net.py <receiver_host> <receiver_port> [out_dir]
       (receiver: measure_net_recv.py on the far host, started first)
"""

import json
import socket
import statistics
import sys
import time
from pathlib import Path

from measure import Sampler, device_string, dist

CHUNK = 64 * 1024
#: (name, target byte/s or None for unthrottled or 0 for idle, seconds)
PHASES = [
    ("idle_pre", 0, 20.0),
    ("full_rate", None, 20.0),
    ("50mbps", 50e6 / 8, 20.0),
    ("10mbps", 10e6 / 8, 20.0),
    ("idle_post", 0, 15.0),
]


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "results")

    sock = socket.create_connection((host, port), timeout=20)
    payload = b"\x00" * CHUNK

    sampler = Sampler()
    sampler.start()
    boundaries = []
    for name, rate, seconds in PHASES:
        start = time.monotonic()
        sent = 0
        if rate == 0:
            time.sleep(seconds)
        else:
            chunks = 0
            while time.monotonic() - start < seconds:
                sock.sendall(payload)
                sent += CHUNK
                chunks += 1
                if rate is not None:
                    delay = (start + chunks * CHUNK / rate) - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
        wall = time.monotonic() - start
        boundaries.append(
            {
                "phase": name,
                "t_start": round(start, 4),
                "t_end": round(start + wall, 4),
                "bytes_sent": sent,
                "achieved_mbps": round(sent * 8 / wall / 1e6, 2),
            }
        )
        mbps = boundaries[-1]["achieved_mbps"]
        print(f"phase {name}: {sent / 1e6:.1f} MB in {wall:.1f}s ({mbps} Mbps)")
    sampler.halt()
    sock.close()

    phases_out = []
    idle_means = []
    for b in boundaries:
        window = [
            s["total_w"] for s in sampler.samples if b["t_start"] + 1.0 <= s["t"] <= b["t_end"]
        ]
        entry = dict(b)
        entry["power_w"] = dist(window)
        phases_out.append(entry)
        if b["phase"].startswith("idle"):
            idle_means.append(statistics.fmean(window))
    idle_w = statistics.fmean(idle_means)

    for entry in phases_out:
        if entry["bytes_sent"] > 0:
            delta_w = entry["power_w"]["mean"] - idle_w
            entry["delta_w_vs_idle"] = round(delta_w, 3)
            duration = entry["t_end"] - entry["t_start"]
            entry["j_per_mb"] = round(max(0.0, delta_w) * duration / (entry["bytes_sent"] / 1e6), 4)

    deltas = [e["delta_w_vs_idle"] for e in phases_out if "delta_w_vs_idle" in e]
    idle_std = max(e["power_w"]["std"] for e in phases_out if e["phase"].startswith("idle"))
    visible = any(d > 2 * idle_std and d > 0.2 for d in deltas)

    record = {
        "kind": "net_tx",
        "device": device_string(),
        "receiver": f"{host}:{port}",
        "phases": phases_out,
        "idle_w": round(idle_w, 3),
        "visibility": {
            "visible_on_rails": visible,
            "criterion": "any TX-phase mean delta > max(2 x idle std, 0.2 W)",
            "max_idle_std_w": round(idle_std, 3),
        },
        "provenance": (
            "measured: TCP stream to a discard receiver on the lab LAN; devkit NIC, NOT a "
            "UAV radio — J/MB here documents the gap to the configured 60 J/MB stress value "
            "and never replaces it; rails sampled ~20 Hz, per-phase mean minus in-run idle"
        ),
        "raw_samples": sampler.samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "net_tx.json").write_text(json.dumps(record, indent=1))
    verdict = "VISIBLE" if visible else "NOT measurable with this instrumentation"
    print(f"net_tx: idle={idle_w:.2f}W deltas={deltas} -> {verdict}")


if __name__ == "__main__":
    main()
