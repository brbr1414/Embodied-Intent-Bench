"""Server-side latency profile, measured on the development Mac as a STAND-IN server.

What the benchmark's remote paths make a server do, timed on this machine:

- full-model inference (the raw-remote backend): batch 1/2/4/8 for the batchable
  semantic models, list-batch for Mask R-CNN — the batch column exists to ground a
  future contention model's capacity parameter (throughput), not because one UAV
  batches its own requests;
- split TAIL execution (the presplit kinds' server half): payload decode + the
  remaining network, via the real backends' ``complete`` seam — the same code a
  mission runs.

Honesty: this machine is an Apple-Silicon laptop (MPS, fp32), a development stand-in,
NOT a deployment server — the table's role is to give the coming server model a
measured shape (relative costs, batch scaling), and every record carries that label.
Energy is not measurable here (and server energy is out of scope by design). Entropy
decode time is content-dependent; tails are timed on one fixed seeded payload and the
payload size is recorded.

Run inside the full-stack venv:

    ~/.venvs/sc2bench/bin/python -m experiments.server_profile.measure_mac_server
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

WARMUP = 5
MIN_ITERS = 30
MIN_WALL_S = 5.0

CKPT_ROOT = Path("~/.cache/sc2-benchmark/resource/ckpt/pascal_voc2012").expanduser()
ES_CKPT = (
    CKPT_ROOT
    / "supervised_compression/entropic_student"
    / "pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta{beta}_from_deeplabv3_resnet50.pt"
)
GHND_CKPT = (
    CKPT_ROOT
    / "supervised_compression/ghnd-bq"
    / "pascal_voc2012-deeplabv3_resnet50-bq3ch_from_deeplabv3_resnet50.pt"
)

OUT_DIR = Path("results/server_profile")


def timed(step, label: str) -> dict:
    for _ in range(WARMUP):
        step()
    laps: list[float] = []
    started = time.perf_counter()
    while len(laps) < MIN_ITERS or (time.perf_counter() - started) < MIN_WALL_S:
        t0 = time.perf_counter()
        step()
        laps.append(time.perf_counter() - t0)
    laps_sorted = sorted(laps)
    record = {
        "label": label,
        "iters": len(laps),
        "latency_ms_mean": statistics.fmean(laps) * 1000.0,
        "latency_ms_std": statistics.stdev(laps) * 1000.0,
        "latency_ms_p95": laps_sorted[int(0.95 * (len(laps) - 1))] * 1000.0,
    }
    print(
        f"{label}: {record['latency_ms_mean']:.1f} ms ±{record['latency_ms_std']:.1f} "
        f"(p95 {record['latency_ms_p95']:.1f}, n={record['iters']})"
    )
    return record


def full_model_cells(device: str) -> list[dict]:
    import torch
    from torchvision.models import get_model, get_model_weights

    cells = []
    for model_id, (width, height) in (
        ("deeplabv3_resnet50", (768, 576)),
        ("lraspp_mobilenet_v3_large", (512, 384)),
    ):
        model = get_model(model_id, weights=get_model_weights(model_id).DEFAULT)
        model = model.eval().to(device)
        for batch in (1, 2, 4, 8):
            torch.manual_seed(0)
            frame = torch.rand(batch, 3, height, width, device=device)

            def step(model=model, frame=frame) -> None:
                with torch.no_grad():
                    model(frame)
                if device == "mps":
                    torch.mps.synchronize()

            record = timed(step, f"{model_id}@{width}x{height} batch{batch}")
            record.update(
                {
                    "kind": "full_model",
                    "model": model_id,
                    "resolution": f"{width}x{height}",
                    "batch": batch,
                    "device": device,
                    "throughput_fps": batch / (record["latency_ms_mean"] / 1000.0),
                }
            )
            cells.append(record)
        del model
    # Mask R-CNN takes a list of images; "batch" is the list length.
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

    mask = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=384, max_size=512
    )
    mask = mask.eval().to(device)
    for batch in (1, 4):
        torch.manual_seed(0)
        frames = [torch.rand(3, 384, 512, device=device) for _ in range(batch)]

        def step(frames=frames) -> None:
            with torch.no_grad():
                mask(frames)
            if device == "mps":
                torch.mps.synchronize()

        record = timed(step, f"maskrcnn_resnet50_fpn@512x384 batch{batch}")
        record.update(
            {
                "kind": "full_model",
                "model": "maskrcnn_resnet50_fpn",
                "resolution": "512x384",
                "batch": batch,
                "device": device,
                "throughput_fps": batch / (record["latency_ms_mean"] / 1000.0),
            }
        )
        cells.append(record)
    return cells


def _spec(config_id: str, parameters: dict):
    from aerointentbench.v2.scenario import ExecutorConfigSpec

    return ExecutorConfigSpec(
        config_id=config_id,
        model_strategy_id=config_id,
        kind="pretrained_split",
        mission_latency_s=0.4,
        energy_j_per_call=8.0,
        communication_mb_per_call=0.1,
        quality_tier=1,
        parameters=parameters,
    )


def seeded_rgb(width: int = 512, height: int = 384) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def tail_cells(device: str) -> list[dict]:
    """Split tails through the real backends' encode->complete seam."""
    from aerointentbench.v2.instance_models import FcmMaskRcnnSplitModel
    from aerointentbench.v2.presplit import _Sc2EntropicStudentModel, _Sc2GhndBqModel

    rgb = seeded_rgb()
    cells = []
    tails = []
    for beta in ("0.64", "5.12"):
        tails.append(
            (
                f"es_b{beta.replace('.', '')}_tail",
                _Sc2EntropicStudentModel(
                    _spec(
                        f"srv_es_{beta}",
                        {
                            "split_backend": "sc2_entropic_student",
                            "checkpoint_path": str(ES_CKPT).format(beta=beta),
                            "input_width_px": 512,
                            "input_height_px": 384,
                            "device": device,
                        },
                    )
                ),
            )
        )
    tails.append(
        (
            "ghnd_bq3_tail",
            _Sc2GhndBqModel(
                _spec(
                    "srv_ghnd",
                    {
                        "split_backend": "sc2_ghnd_bq",
                        "checkpoint_path": str(GHND_CKPT),
                        "input_width_px": 512,
                        "input_height_px": 384,
                        "device": device,
                        "bottleneck_channels": 3,
                    },
                )
            ),
        )
    )
    tails.append(
        (
            "fcm_maskrcnn_tail",
            FcmMaskRcnnSplitModel(
                _spec(
                    "srv_fcm",
                    {
                        "split_backend": "fcm_maskrcnn_fpn",
                        "score_threshold": 0.5,
                        "input_min_size_px": 384,
                        "input_max_size_px": 512,
                        "device": device,
                        "feature_dtype": "uint8",
                    },
                )
            ),
        )
    )
    for name, model in tails:
        payload, payload_mb, _ = model.encode(rgb)

        def step(model=model, payload=payload) -> None:
            model.complete(payload, rgb.shape)

        record = timed(step, f"{name} (payload {payload_mb * 1000.0:.1f} KB)")
        record.update(
            {
                "kind": "split_tail",
                "model": name,
                "resolution": "512x384",
                "batch": 1,
                "device": device,
                "payload_kb": payload_mb * 1000.0,
                "throughput_fps": 1.0 / (record["latency_ms_mean"] / 1000.0),
            }
        )
        cells.append(record)
    return cells


def main() -> int:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"stand-in server: {platform.platform()} / torch {torch.__version__} / {device}")
    cells = full_model_cells(device)
    # sc2bench tails: try the accelerator, fall back to CPU (the entropy stage is
    # CPU-side either way); record which device actually ran.
    try:
        cells += tail_cells(device)
    except Exception as error:
        print(f"tails on {device} failed ({type(error).__name__}: {error}); retrying on cpu")
        cells += tail_cells("cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mac_server_profile.json"
    out.write_text(
        json.dumps(
            {
                "provenance": (
                    "development-machine STAND-IN server: Apple-Silicon laptop, fp32, "
                    "software stack only — NOT a deployment server; latencies exist to "
                    "shape the coming server model (relative costs, batch scaling) and "
                    "must never be quoted as production server performance. Energy not "
                    "measurable here. Tail timings use one fixed seeded payload; entropy "
                    "decode time is content-dependent (payload size recorded)."
                ),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cells": cells,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
