#!/usr/bin/env python3
"""One (model, power-mode, repeat) energy/latency measurement on a Jetson board.

Power: every labelled INA3221 module rail on the board, sampled in a background thread
at ~20 Hz. Rail sets differ per board and are discovered, not assumed — AGX Orin exposes
one chip at i2c 1-0040 (VDD_GPU_SOC / VDD_CPU_CV / VIN_SYS_5V0), AGX Xavier two chips at
1-0040 + 1-0041 (GPU / CPU / SOC + CV / VDDRQ / SYS5V); the rails actually summed are
recorded in each output file. Protocol: 10 s idle baseline, 5 warmup inferences, then
CUDA-synchronised timed inferences until BOTH >=30 iterations AND >=10 s of wall time.
Energy per inference is reported two ways: total (mean run power x wall / iters) and
marginal (subtracting the same-invocation idle mean). Raw samples are saved so the
aggregation can be redone later.

Usage: measure.py <model_id> <mode_label> <repeat_index> [out_dir] [WxH]

The optional trailing argument selects the input resolution (default 512x384). Runs at a
non-default resolution carry the resolution in the output filename so they never collide
with the original sweep's records.
"""

import json
import statistics
import sys
import threading
import time
from pathlib import Path

HWMON_GLOB = "/sys/bus/i2c/drivers/ina3221/1-004*/hwmon/hwmon*"
MIN_ITERS = 30
MIN_WALL_S = 10.0
IDLE_S = 10.0
WARMUP = 5


def rails():
    import glob

    found = []
    for h in sorted(Path(p) for p in glob.glob(HWMON_GLOB)):
        for i in (1, 2, 3):
            label = h / f"in{i}_label"
            if label.exists():
                found.append((label.read_text().strip(), h / f"in{i}_input", h / f"curr{i}_input"))
    if not found:
        raise SystemExit(f"no readable INA3221 rails under {HWMON_GLOB}")
    return found


def device_string():
    import torch

    model = Path("/proc/device-tree/model").read_text().strip("\x00").strip()
    parts = Path("/etc/nv_tegra_release").read_text().split(",")
    l4t = f"{parts[0].split()[1]}.{parts[1].split(':')[1].strip()}"
    return f"{model} (L4T {l4t}, torch {torch.__version__})"


class Sampler(threading.Thread):
    def __init__(self, hz=20.0):
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        self.samples = []
        self._halted = False  # NB: must not shadow threading.Thread's internal _stop
        self._rails = rails()

    def run(self):
        while not self._halted:
            t = time.monotonic()
            total = 0.0
            per = {}
            for name, vpath, cpath in self._rails:
                watts = int(vpath.read_text()) * int(cpath.read_text()) / 1e6
                per[name] = round(watts, 3)
                total += watts
            self.samples.append({"t": round(t, 4), "total_w": round(total, 3), **per})
            delay = self.period - (time.monotonic() - t)
            if delay > 0:
                time.sleep(delay)

    def halt(self):
        self._halted = True
        self.join()


def dist(values):
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.fmean(s),
        "median": s[n // 2],
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p95": s[max(0, round(0.95 * (n - 1)))],
    }


def main():
    model_id, mode_label, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_dir = Path(sys.argv[4] if len(sys.argv) > 4 else "results")
    resolution = sys.argv[5] if len(sys.argv) > 5 else "512x384"
    width, height = (int(v) for v in resolution.split("x"))

    import torch
    from torchvision.models import get_model, get_model_weights

    device = torch.device("cuda")
    torch.manual_seed(0)
    frame = torch.rand(1, 3, height, width, device=device)

    idle_sampler = Sampler()
    idle_sampler.start()
    time.sleep(IDLE_S)
    idle_sampler.halt()
    idle_power = [s["total_w"] for s in idle_sampler.samples]

    weights = get_model_weights(model_id).DEFAULT
    model = get_model(model_id, weights=weights).eval().to(device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(frame)
        torch.cuda.synchronize()

        run_sampler = Sampler()
        run_sampler.start()
        latencies = []
        wall_start = time.monotonic()
        while len(latencies) < MIN_ITERS or (time.monotonic() - wall_start) < MIN_WALL_S:
            t0 = time.monotonic()
            model(frame)
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - t0)
        wall = time.monotonic() - wall_start
        run_sampler.halt()

    run_power = [s["total_w"] for s in run_sampler.samples]
    idle_mean = statistics.fmean(idle_power)
    run_mean = statistics.fmean(run_power)
    iters = len(latencies)

    rail_names = [r[0] for r in rails()]
    record = {
        "device": device_string(),
        "model": model_id,
        "mode": mode_label,
        "repeat": repeat,
        "resolution": resolution,
        "input": f"1x3x{height}x{width} uniform random, seed 0",
        "iters": iters,
        "wall_s": round(wall, 4),
        "latency_s": dist(latencies),
        "power_idle_w": dist(idle_power),
        "power_run_w": dist(run_power),
        "energy_total_j_per_inf": run_mean * wall / iters,
        "energy_marginal_j_per_inf": max(0.0, run_mean - idle_mean) * wall / iters,
        "rails": rail_names,
        "provenance": (
            f"measured: INA3221 module rails ({'+'.join(rail_names)}) at ~20 Hz; "
            "energy = mean power x wall / iterations; marginal subtracts same-invocation idle; "
            "default DVFS governor (jetson_clocks NOT applied)"
        ),
        "raw_idle_samples": idle_sampler.samples,
        "raw_run_samples": run_sampler.samples,
        "raw_latencies_s": [round(v, 5) for v in latencies],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    res_tag = "" if resolution == "512x384" else f"__{resolution}"
    path = out_dir / f"{mode_label}__{model_id}{res_tag}__rep{repeat}.json"
    path.write_text(json.dumps(record, indent=1))
    print(
        f"{mode_label} {model_id} {resolution} rep{repeat}: "
        f"lat={record['latency_s']['mean'] * 1000:.1f}ms "
        f"run={run_mean:.2f}W idle={idle_mean:.2f}W "
        f"E_total={record['energy_total_j_per_inf']:.3f}J "
        f"E_marg={record['energy_marginal_j_per_inf']:.3f}J"
    )


if __name__ == "__main__":
    main()
