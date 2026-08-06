#!/usr/bin/env python3
"""Thermal-soak drift under sustained inference (plan item A6).

Missions run 900 s; the 10 s sweep cells never reach thermal equilibrium. This runs
one model back-to-back (no cadence gaps — worst case) for ``duration_s``, logging a
snapshot every ~10 s: recent latency, thermal zones, recent rail power. The summary
compares the first and last minute so late-mission latency drift is a number, not a
guess.

Usage: measure_thermal.py <model_id> <duration_s> [out_dir] [mode_label]
"""

import glob
import json
import statistics
import sys
import time
from pathlib import Path

from measure import Sampler, device_string, dist


def thermal_zones():
    zones = {}
    for zdir in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        z = Path(zdir)
        try:
            zones[z.joinpath("type").read_text().strip()] = (
                int(z.joinpath("temp").read_text()) / 1000.0
            )
        except (OSError, ValueError):
            continue
    return zones


def main():
    model_id = sys.argv[1]
    duration_s = float(sys.argv[2])
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "results")
    mode_label = sys.argv[4] if len(sys.argv) > 4 else "MODE_UNKNOWN"

    import torch
    from torchvision.models import get_model, get_model_weights

    torch.manual_seed(0)
    frame = torch.rand(1, 3, 384, 512, device="cuda")
    weights = get_model_weights(model_id).DEFAULT
    model = get_model(model_id, weights=weights).eval().to("cuda")
    with torch.inference_mode():
        for _ in range(5):
            model(frame)
        torch.cuda.synchronize()

    sampler = Sampler()
    sampler.start()
    latencies = []
    stamps = []
    snapshots = [{"t_s": 0.0, "zones_c": thermal_zones()}]
    start = time.monotonic()
    next_snap = 10.0
    with torch.inference_mode():
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_s:
                break
            t0 = time.monotonic()
            model(frame)
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - t0)
            stamps.append(elapsed)
            if elapsed >= next_snap:
                # zip strict= would need 3.10; this runs on the board's 3.8
                window = [v for v, t in zip(latencies, stamps) if t > elapsed - 10]  # noqa: B905
                snapshots.append(
                    {
                        "t_s": round(elapsed, 1),
                        "latency_ms_last10s": round(statistics.fmean(window) * 1000, 2),
                        "zones_c": thermal_zones(),
                    }
                )
                next_snap += 10.0
    wall = time.monotonic() - start
    sampler.halt()

    def minute(lo, hi):
        vals = [v for v, t in zip(latencies, stamps) if lo <= t < hi]  # noqa: B905
        return dist(vals) if vals else None

    first_min = minute(0.0, 60.0)
    last_min = minute(wall - 60.0, wall)
    power = [s["total_w"] for s in sampler.samples]
    zone_series = {}
    for snap in snapshots:
        for name, temp in snap["zones_c"].items():
            zone_series.setdefault(name, []).append(temp)

    record = {
        "kind": "thermal_soak",
        "device": device_string(),
        "model": model_id,
        "mode": mode_label,
        "input": "1x3x384x512 uniform random, seed 0",
        "duration_s": round(wall, 1),
        "inferences": len(latencies),
        "latency_first_minute_s": first_min,
        "latency_last_minute_s": last_min,
        "drift_pct": round((last_min["mean"] / first_min["mean"] - 1) * 100, 2)
        if first_min and last_min
        else None,
        "power_w": dist(power),
        "zones_start_c": snapshots[0]["zones_c"],
        "zones_end_c": snapshots[-1]["zones_c"],
        "zones_max_c": {name: max(vals) for name, vals in zone_series.items()},
        "snapshots": snapshots,
        "provenance": (
            "measured: back-to-back inference (no gaps — worst case vs any real cadence), "
            "default DVFS; thermal zones from /sys/class/thermal; rails ~20 Hz"
        ),
        "raw_latencies_s": [round(v, 5) for v in latencies],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"thermal__{mode_label}__{model_id}.json").write_text(json.dumps(record, indent=1))
    print(
        f"thermal {mode_label} {model_id}: {len(latencies)} inf / {wall:.0f}s, "
        f"first-min={first_min['mean'] * 1000:.1f}ms last-min={last_min['mean'] * 1000:.1f}ms "
        f"drift={record['drift_pct']}% max_temp={max(record['zones_max_c'].values()):.1f}C"
    )


if __name__ == "__main__":
    main()
