"""Aggregate catalog-workload sweep records into per-(mode, workload) statistics.

Input: the per-invocation JSONs written by ``measure_catalog.py`` (one file per
mode x workload x repeat). Same restatement rule as aggregate.py: nothing is
re-measured. Split-head workloads additionally report the measured wire payload
(bytes the head actually encoded, content-dependent — on the sweep's random
input, not mission frames).

    python -m experiments.jetson_power.aggregate_catalog <results_dir> [...]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODE_ORDER = [
    "MODE_10W",
    "MODE_15W",
    "MODE_15W_DESKTOP",
    "MODE_30W_2CORE",
    "MODE_30W_4CORE",
    "MODE_30W_6CORE",
    "MODE_30W_ALL",
    "MAXN",
]
WORKLOAD_ORDER = [
    "lraspp_fp32",
    "lraspp_fp16",
    "dlv3_fp32",
    "dlv3_fp16",
    "dlv3_fp32_512",
    "dlv3_fp16_512",
    "lraspp_fp32_768",
    "lraspp_fp16_768",
    "es_b064",
    "es_b512",
    "ghnd_bq3",
    "fcm_head",
    "maskrcnn",
]


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main(argv: list[str]) -> int:
    records = []
    for root in argv:
        for path in sorted(Path(root).rglob("*__rep*.json")):
            records.append(json.loads(path.read_text()))
    if not records:
        print("no records found", file=sys.stderr)
        return 2

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["workload"])].append(record)

    out: dict[str, dict] = {}
    for (mode, workload), reps in sorted(grouped.items()):
        pooled_lat = sorted(v for r in reps for v in r["raw_latencies_s"])
        payload = [r["payload_bytes"]["mean"] for r in reps if r.get("payload_bytes")]
        out[f"{mode}/{workload}"] = {
            "mode": mode,
            "workload": workload,
            "resolution": reps[0]["resolution"],
            "repeats": len(reps),
            "iters_total": sum(r["iters"] for r in reps),
            "latency_ms": mean_std([r["latency_s"]["mean"] * 1e3 for r in reps]),
            "latency_p95_ms_pooled": pooled_lat[int(0.95 * (len(pooled_lat) - 1))] * 1e3,
            "power_idle_w": mean_std([r["power_idle_w"]["mean"] for r in reps]),
            "power_run_w": mean_std([r["power_run_w"]["mean"] for r in reps]),
            "energy_total_j": mean_std([r["energy_total_j_per_inf"] for r in reps]),
            "energy_marginal_j": mean_std([r["energy_marginal_j_per_inf"] for r in reps]),
            "payload_kb": mean_std([v / 1e3 for v in payload]) if payload else None,
        }

    device = records[0]["device"]
    provenance = records[0]["provenance"]

    print(f"# Jetson catalog sweep — {device}\n")
    print(f"{provenance}\n")
    print(
        "| mode | workload | input | latency ms (±std) | p95 ms | idle W | run W | "
        "E_total J/inf (±) | E_marginal J/inf (±) | payload KB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mode in MODE_ORDER:
        for workload in WORKLOAD_ORDER:
            entry = out.get(f"{mode}/{workload}")
            if not entry:
                continue
            lat, run = entry["latency_ms"], entry["power_run_w"]
            et, em = entry["energy_total_j"], entry["energy_marginal_j"]
            payload = entry["payload_kb"]
            payload_cell = f"{payload['mean']:.1f}" if payload else "—"
            print(
                f"| {mode} | {workload} | {entry['resolution']} | "
                f"{lat['mean']:.1f} ±{lat['std']:.1f} | "
                f"{entry['latency_p95_ms_pooled']:.1f} | "
                f"{entry['power_idle_w']['mean']:.2f} | {run['mean']:.2f} | "
                f"{et['mean']:.3f} ±{et['std']:.3f} | {em['mean']:.3f} ±{em['std']:.3f} | "
                f"{payload_cell} |"
            )

    summary_path = Path(argv[0]).parent / "jetson_catalog_summary.json"
    summary_path.write_text(
        json.dumps({"device": device, "provenance": provenance, "cells": out}, indent=1)
    )
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
