#!/usr/bin/env python3
"""Energy per mission-second at a fixed inference cadence (plan item A4).

The benchmark's battery model charges per-call joules only; this measures what a
mission actually pays per second of wall time when one inference is triggered every
``cadence_s`` seconds (race-to-idle between calls, default DVFS). The interesting
comparison is against the pure-idle window (``model_id`` = ``none``): the delta shows
how much duty-cycling can save at all on top of the board's idle floor.

Reuses the shared Sampler / protocol pieces from ``measure.py`` (same directory).

Usage: measure_cadence.py <model_id|none> <cadence_s> <duration_s> [out_dir] [mode_label]
"""

import json
import statistics
import sys
import time
from pathlib import Path

from measure import IDLE_S, Sampler, device_string, dist


def main():
    model_id = sys.argv[1]
    cadence_s = float(sys.argv[2])
    duration_s = float(sys.argv[3])
    out_dir = Path(sys.argv[4] if len(sys.argv) > 4 else "results")
    mode_label = sys.argv[5] if len(sys.argv) > 5 else "MODE_UNKNOWN"

    import torch

    idle_sampler = Sampler()
    idle_sampler.start()
    time.sleep(IDLE_S)
    idle_sampler.halt()
    idle_power = [s["total_w"] for s in idle_sampler.samples]

    model = None
    frame = None
    if model_id != "none":
        from torchvision.models import get_model, get_model_weights

        torch.manual_seed(0)
        frame = torch.rand(1, 3, 384, 512, device="cuda")
        weights = get_model_weights(model_id).DEFAULT
        model = get_model(model_id, weights=weights).eval().to("cuda")
        with torch.inference_mode():
            for _ in range(5):
                model(frame)
            torch.cuda.synchronize()
        time.sleep(2.0)  # settle back toward idle before the window starts

    run_sampler = Sampler()
    run_sampler.start()
    latencies = []
    start = time.monotonic()
    if model is None:
        time.sleep(duration_s)
    else:
        slots = 0
        with torch.inference_mode():
            while time.monotonic() - start < duration_s:
                t0 = time.monotonic()
                model(frame)
                torch.cuda.synchronize()
                latencies.append(time.monotonic() - t0)
                slots += 1
                delay = (start + slots * cadence_s) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    wall = time.monotonic() - start
    run_sampler.halt()

    run_power = [s["total_w"] for s in run_sampler.samples]
    idle_mean = statistics.fmean(idle_power)
    run_mean = statistics.fmean(run_power)

    record = {
        "kind": "cadence",
        "device": device_string(),
        "model": model_id,
        "mode": mode_label,
        "cadence_s": cadence_s,
        "window_s": round(wall, 3),
        "inferences": len(latencies),
        "input": "1x3x384x512 uniform random, seed 0" if model_id != "none" else "none",
        "latency_s": dist(latencies) if latencies else None,
        "power_idle_w": dist(idle_power),
        "power_run_w": dist(run_power),
        "energy_window_j": run_mean * wall,
        "energy_per_mission_second_j": run_mean,
        "marginal_energy_per_mission_second_j": max(0.0, run_mean - idle_mean),
        "provenance": (
            "measured: one inference every cadence_s seconds, race-to-idle between calls, "
            "default DVFS; energy over the whole window from all INA3221 module rails at "
            "~20 Hz; 'none' = pure idle window with the same protocol"
        ),
        "raw_idle_samples": idle_sampler.samples,
        "raw_run_samples": run_sampler.samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "none" if model_id == "none" else f"{model_id}__c{cadence_s:g}"
    path = out_dir / f"cadence__{mode_label}__{tag}.json"
    path.write_text(json.dumps(record, indent=1))
    print(
        f"cadence {mode_label} {model_id} c={cadence_s:g}s: {len(latencies)} inf in {wall:.1f}s, "
        f"run={run_mean:.2f}W idle={idle_mean:.2f}W "
        f"E/s={record['energy_per_mission_second_j']:.2f}J "
        f"marginal/s={record['marginal_energy_per_mission_second_j']:.2f}J"
    )


if __name__ == "__main__":
    main()
