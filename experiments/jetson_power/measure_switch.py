#!/usr/bin/env python3
"""Model-switch cost on Jetson: cold load, cold-start inference, resident set (A2/A3).

Two invocation modes, run on the device:

    measure_switch.py cold <model_id> [out_dir]
        Fresh-process timeline for one model: torch import, CUDA context init,
        checkpoint construction+load (weights already in the local torchvision cache,
        so this is disk-cache-warm, download excluded), transfer to device, then the
        first / second / steady-state inference. The first inference is the number the
        sweep's warmup deliberately discarded (A3): what a cold local fallback would
        really pay.

    measure_switch.py resident [out_dir]
        Loads all six zoo models into one process sequentially, recording per-model
        load time and the cumulative device-memory / RSS footprint (unified memory:
        CUDA allocator bytes and process RSS overlap and are both reported). Then a
        round-robin warm-swap pass (one inference per model per cycle, 5 cycles)
        versus a dedicated steady pass (10 consecutive inferences per model) shows
        what switching between resident models costs at inference time.

No power sampling here — these are latency/footprint measurements. Input is the
sweep's 512x384 uniform-random frame for comparability.
"""

import json
import statistics
import sys
import time
from pathlib import Path

MODELS = [
    "lraspp_mobilenet_v3_large",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "fcn_resnet101",
    "deeplabv3_resnet101",
]


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
    }


def rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def cold(model_id, out_dir):
    t0 = time.perf_counter()
    import torch

    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    t_cuda_init = time.perf_counter() - t0

    from torchvision.models import get_model, get_model_weights

    torch.manual_seed(0)
    frame = torch.rand(1, 3, 384, 512, device="cuda")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    weights = get_model_weights(model_id).DEFAULT
    model = get_model(model_id, weights=weights).eval()
    t_construct_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = model.to("cuda")
    torch.cuda.synchronize()
    t_to_device = time.perf_counter() - t0

    with torch.inference_mode():
        t0 = time.perf_counter()
        model(frame)
        torch.cuda.synchronize()
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        model(frame)
        torch.cuda.synchronize()
        t_second = time.perf_counter() - t0

        steady = []
        for _ in range(10):
            t0 = time.perf_counter()
            model(frame)
            torch.cuda.synchronize()
            steady.append(time.perf_counter() - t0)

    record = {
        "kind": "switch_cold",
        "model": model_id,
        "input": "1x3x384x512 uniform random, seed 0",
        "t_torch_import_s": round(t_import, 4),
        "t_cuda_init_s": round(t_cuda_init, 4),
        "t_construct_load_s": round(t_construct_load, 4),
        "t_to_device_s": round(t_to_device, 4),
        "t_first_inference_s": round(t_first, 4),
        "t_second_inference_s": round(t_second, 4),
        "steady_inference_s": dist(steady),
        "rss_mb_after": round(rss_mb(), 1),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "provenance": (
            "measured: fresh process per model; weights already in the local torchvision "
            "cache (download time excluded); default DVFS"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"switch_cold__{model_id}.json").write_text(json.dumps(record, indent=1))
    print(
        f"cold {model_id}: import={t_import:.2f}s cuda={t_cuda_init:.2f}s "
        f"load={t_construct_load:.2f}s to_dev={t_to_device:.2f}s "
        f"first={t_first * 1000:.0f}ms second={t_second * 1000:.0f}ms "
        f"steady={record['steady_inference_s']['mean'] * 1000:.1f}ms"
    )


def resident(out_dir):
    import torch
    from torchvision.models import get_model, get_model_weights

    torch.manual_seed(0)
    frame = torch.rand(1, 3, 384, 512, device="cuda")
    torch.cuda.synchronize()

    loaded = {}
    load_log = []
    for model_id in MODELS:
        t0 = time.perf_counter()
        weights = get_model_weights(model_id).DEFAULT
        model = get_model(model_id, weights=weights).eval().to("cuda")
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        with torch.inference_mode():
            t0 = time.perf_counter()
            model(frame)
            torch.cuda.synchronize()
            first_s = time.perf_counter() - t0
        loaded[model_id] = model
        load_log.append(
            {
                "model": model_id,
                "load_to_device_s": round(load_s, 4),
                "first_inference_s": round(first_s, 4),
                "cumulative_cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                "cumulative_cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
                "cumulative_rss_mb": round(rss_mb(), 1),
            }
        )
        print(f"resident +{model_id}: {load_log[-1]}")

    swap = {model_id: [] for model_id in MODELS}
    with torch.inference_mode():
        for _ in range(5):
            for model_id in MODELS:
                t0 = time.perf_counter()
                loaded[model_id](frame)
                torch.cuda.synchronize()
                swap[model_id].append(time.perf_counter() - t0)

        dedicated = {}
        for model_id in MODELS:
            runs = []
            for _ in range(10):
                t0 = time.perf_counter()
                loaded[model_id](frame)
                torch.cuda.synchronize()
                runs.append(time.perf_counter() - t0)
            dedicated[model_id] = dist(runs)

    record = {
        "kind": "switch_resident",
        "input": "1x3x384x512 uniform random, seed 0",
        "load_sequence": load_log,
        "round_robin_inference_s": {m: dist(v) for m, v in swap.items()},
        "dedicated_inference_s": dedicated,
        "provenance": (
            "measured: all six models resident in one process (unified memory); round-robin = "
            "one inference per model per cycle x 5 cycles; dedicated = 10 consecutive; "
            "default DVFS"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "switch_resident.json").write_text(json.dumps(record, indent=1))
    for model_id in MODELS:
        rr, ded = record["round_robin_inference_s"][model_id], dedicated[model_id]
        print(
            f"swap {model_id}: round-robin={rr['mean'] * 1000:.1f}ms "
            f"dedicated={ded['mean'] * 1000:.1f}ms"
        )


def main():
    mode = sys.argv[1]
    if mode == "cold":
        cold(sys.argv[2], Path(sys.argv[3] if len(sys.argv) > 3 else "results"))
    elif mode == "resident":
        resident(Path(sys.argv[2] if len(sys.argv) > 2 else "results"))
    else:
        raise SystemExit(f"unknown mode {mode!r} (expected 'cold' or 'resident')")


if __name__ == "__main__":
    main()
