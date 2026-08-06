"""Aggregate Jetson power-sweep records into per-(mode, model, resolution) statistics.

Input: the per-invocation JSONs written by ``measure.py`` (one file per
mode x model x resolution x repeat, raw 20 Hz power samples and per-inference
latencies included). Records written before the resolution argument existed carry
no ``resolution`` field; it is recovered from their ``input`` string. Aggregation
reports, across the repeats of each combination:

- latency: mean of per-repeat means, std across repeats, pooled p95
- power: idle and run means (std across repeats)
- energy per inference: total and marginal (mean +- std across repeats)

Everything here restates the measurement files; nothing is re-measured.

    python -m experiments.jetson_power.aggregate <results_dir> [<results_dir> ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODE_ORDER = ["MODE_10W", "MODE_15W", "MODE_30W", "MODE_30W_ALL", "MODE_50W", "MAXN"]
MODEL_ORDER = [
    "lraspp_mobilenet_v3_large",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "fcn_resnet101",
    "deeplabv3_resnet101",
]


def resolution_of(record: dict) -> str:
    if "resolution" in record:
        return record["resolution"]
    dims = record["input"].split()[0].split("x")  # "1x3x<H>x<W>"
    return f"{dims[3]}x{dims[2]}"


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

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["model"], resolution_of(record))].append(record)

    out: dict[str, dict] = {}
    for (mode, model, resolution), reps in sorted(grouped.items()):
        pooled_lat = [v for r in reps for v in r["raw_latencies_s"]]
        pooled_lat.sort()
        out[f"{mode}/{model}/{resolution}"] = {
            "mode": mode,
            "model": model,
            "resolution": resolution,
            "repeats": len(reps),
            "iters_total": sum(r["iters"] for r in reps),
            "latency_ms": mean_std([r["latency_s"]["mean"] * 1e3 for r in reps]),
            "latency_p95_ms_pooled": pooled_lat[int(0.95 * (len(pooled_lat) - 1))] * 1e3,
            "power_idle_w": mean_std([r["power_idle_w"]["mean"] for r in reps]),
            "power_run_w": mean_std([r["power_run_w"]["mean"] for r in reps]),
            "energy_total_j": mean_std([r["energy_total_j_per_inf"] for r in reps]),
            "energy_marginal_j": mean_std([r["energy_marginal_j_per_inf"] for r in reps]),
        }

    device = records[0]["device"]
    provenance = records[0]["provenance"]
    resolutions = sorted({entry["resolution"] for entry in out.values()})

    print(f"# Jetson power sweep — {device}\n")
    print(f"uniform random input, seed 0; {provenance}\n")
    print(
        "| mode | model | input | latency ms (±std) | p95 ms | idle W | run W | "
        "E_total J/inf (±) | E_marginal J/inf (±) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for mode in MODE_ORDER:
        for model in MODEL_ORDER:
            for resolution in resolutions:
                entry = out.get(f"{mode}/{model}/{resolution}")
                if not entry:
                    continue
                lat, run = entry["latency_ms"], entry["power_run_w"]
                et, em = entry["energy_total_j"], entry["energy_marginal_j"]
                print(
                    f"| {mode} | {model} | {resolution} | {lat['mean']:.1f} ±{lat['std']:.1f} | "
                    f"{entry['latency_p95_ms_pooled']:.1f} | "
                    f"{entry['power_idle_w']['mean']:.2f} | {run['mean']:.2f} | "
                    f"{et['mean']:.3f} ±{et['std']:.3f} | {em['mean']:.3f} ±{em['std']:.3f} |"
                )

    summary_path = Path(argv[0]).parent / "jetson_power_summary.json"
    summary_path.write_text(
        json.dumps({"device": device, "provenance": provenance, "cells": out}, indent=1)
    )
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
