#!/usr/bin/env python3
"""One (catalog workload, power-mode, repeat) energy/latency measurement on a Jetson.

Same protocol as measure.py (10 s idle baseline, 5 warmups, CUDA-synchronised timed
iterations until >=30 iters AND >=10 s, INA3221 rails at ~20 Hz, total + marginal
energy), extended to the model-catalog workloads that measure.py cannot express:

- onboard tiers with an explicit dtype (fp32/fp16) at the catalog resolution;
- Mask R-CNN onboard (list input, detection transform inside the timed region);
- the onboard HEAD of each published split (split_heads.py transcriptions), where
  the timed region is the real head compute: encoder forward + entropy coding /
  quantization — the work that produces the wire payload. Payload bytes per call
  are recorded from what the head actually encoded.

Inputs are seeded uniform-random tensors (same convention as the original sweep):
content-independent compute for the semantic models and heads; for Mask R-CNN the
RPN/ROI stage runs its fixed top-k proposal budget, but detection counts on noise
are near zero — recorded in provenance, do not read its absolute latency as
scene-realistic without a content check.

Usage: measure_catalog.py <workload> <mode_label> <repeat_index> [out_dir]

Workloads: dlv3_fp32 dlv3_fp16 lraspp_fp32 lraspp_fp16 maskrcnn
           es_b064 es_b512 ghnd_bq3 fcm_head
Checkpoint locations come from CKPT_DIR (default /images/aerobench/sc2_ckpt).
Python 3.8 compatible.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

from measure import IDLE_S, MIN_ITERS, MIN_WALL_S, WARMUP, Sampler, device_string, dist, rails

CKPT_DIR = Path(os.environ.get("CKPT_DIR", "/images/aerobench/sc2_ckpt"))
QAIHUB_DIR = Path(os.environ.get("QAIHUB_DIR", "/images/aerobench/qaihub"))

#: onnx workload prefix -> (qaihub model name, resolution WxH). These run the
#: PUBLISHED Qualcomm AI Hub ONNX artifacts (float and pre-quantized w8a8) on the
#: CPU via onnxruntime at ORT_ENABLE_BASIC -- the same execution path the
#: benchmark's onnx_semantic_segmentation kind uses. CPU-bound by construction, so
#: unlike the GPU workloads these genuinely vary across the core-count power modes.
ONNX_MODELS = {
    "onnx_dlv3plus": ("deeplabv3_plus_mobilenet", "520x520"),
    "onnx_segformer": ("segformer_base", "512x512"),
    "onnx_ffnet40s": ("ffnet_40s", "2048x1024"),
}

ES_CKPT = "pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta{beta}_from_deeplabv3_resnet50.pt"
GHND_CKPT = "pascal_voc2012-deeplabv3_resnet50-bq{ch}ch_from_deeplabv3_resnet50.pt"

#: workload -> (resolution WxH, description)
WORKLOADS = {
    "dlv3_fp32": ("768x576", "deeplabv3_resnet50 onboard float32 (catalog local_strong_fp32)"),
    "dlv3_fp16": ("768x576", "deeplabv3_resnet50 onboard float16 (catalog local_strong_fp16)"),
    "lraspp_fp32": (
        "512x384",
        "lraspp_mobilenet_v3_large onboard float32 (catalog local_light_fp32)",
    ),
    "lraspp_fp16": (
        "512x384",
        "lraspp_mobilenet_v3_large onboard float16 (catalog local_light_fp16)",
    ),
    "maskrcnn": (
        "512x384",
        "maskrcnn_resnet50_fpn onboard (catalog maskrcnn_onboard_full; min 384 / max 512)",
    ),
    # Resolution-tier variants (2026-08-21): the input-resolution knob on the SAME
    # checkpoints -- strong at reduced resolution, light at increased resolution.
    "dlv3_fp32_512": ("512x384", "deeplabv3_resnet50 float32 at reduced 512x384"),
    "dlv3_fp16_512": ("512x384", "deeplabv3_resnet50 float16 at reduced 512x384"),
    "lraspp_fp32_768": ("768x576", "lraspp_mobilenet_v3_large float32 at increased 768x576"),
    "lraspp_fp16_768": ("768x576", "lraspp_mobilenet_v3_large float16 at increased 768x576"),
    "onnx_dlv3plus_w8a8": ("520x520", "QAIHub DeepLabV3+-MobileNet w8a8 ONNX (CPU EP)"),
    "onnx_dlv3plus_float": ("520x520", "QAIHub DeepLabV3+-MobileNet float ONNX (CPU EP)"),
    "onnx_segformer_w8a8": ("512x512", "QAIHub SegFormer-B0 ADE w8a8 ONNX (CPU EP)"),
    "onnx_segformer_float": ("512x512", "QAIHub SegFormer-B0 ADE float ONNX (CPU EP)"),
    "onnx_ffnet40s_w8a8": ("2048x1024", "QAIHub FFNet-40S Cityscapes w8a8 ONNX (CPU EP)"),
    "onnx_ffnet40s_float": ("2048x1024", "QAIHub FFNet-40S Cityscapes float ONNX (CPU EP)"),
    "es_b064": ("512x384", "Entropic Student beta0.64 onboard head (catalog presplit_es_b064)"),
    "es_b512": ("512x384", "Entropic Student beta5.12 onboard head (catalog presplit_es_b512)"),
    "ghnd_bq3": ("512x384", "GHND-BQ 3ch onboard head (catalog presplit_ghnd_bq3)"),
    "fcm_head": (
        "512x384",
        "Mask R-CNN backbone+FPN onboard head, uint8 pricing (catalog presplit_maskrcnn_fcm)",
    ),
}


def _normalized_frame(torch, height, width, device):
    """Seeded uniform RGB run through the ES/GHND preprocessing (ImageNet norm)."""
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, height, width, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def build_workload(workload, device):
    """Return (step callable -> payload_bytes or None, input description)."""
    import torch

    resolution, _ = WORKLOADS[workload]
    width, height = (int(v) for v in resolution.split("x"))

    if workload.startswith("onnx_"):
        import numpy as np
        import onnxruntime as ort

        base, precision = workload.rsplit("_", 1)
        model_name, _ = ONNX_MODELS[base]
        folder = f"{model_name}-onnx-{precision}"
        path = QAIHUB_DIR / folder / folder / f"{model_name}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        shape = [1 if isinstance(s, str) else s for s in inp.shape]
        rng = np.random.default_rng(0)
        if "uint8" in inp.type:
            frame_np = rng.integers(0, 256, size=shape, dtype=np.uint8)
        else:
            frame_np = rng.random(shape).astype(np.float32)
        feed = {inp.name: frame_np}

        def step():
            session.run(None, feed)
            return None

        return step, f"{shape} {inp.type} seeded random (ONNX CPU EP, ORT_ENABLE_BASIC)"

    if workload.startswith(("dlv3_", "lraspp_")):
        from torchvision.models import get_model, get_model_weights

        model_id = (
            "deeplabv3_resnet50" if workload.startswith("dlv3_") else "lraspp_mobilenet_v3_large"
        )
        model = get_model(model_id, weights=get_model_weights(model_id).DEFAULT).eval().to(device)
        torch.manual_seed(0)
        frame = torch.rand(1, 3, height, width, device=device)
        if "_fp16" in workload:
            model = model.half()
            frame = frame.half()

        def step():
            model(frame)
            return None

        return step, f"1x3x{height}x{width} uniform random, seed 0"

    if workload == "maskrcnn":
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=384, max_size=512
        )
        model.to(device).eval()
        torch.manual_seed(0)
        frame = torch.rand(3, height, width, device=device)

        def step():
            model([frame])
            return None

        return step, f"3x{height}x{width} uniform random in [0,1], seed 0"

    if workload in ("es_b064", "es_b512"):
        from split_heads import EntropicStudentHead

        beta = "0.64" if workload == "es_b064" else "5.12"
        head = EntropicStudentHead(str(CKPT_DIR / ES_CKPT.format(beta=beta)), device)
        frame = _normalized_frame(torch, height, width, device)

        def step():
            payload_bytes, _ = head.encode(frame)
            return payload_bytes

        return step, f"1x3x{height}x{width} uniform random, seed 0, ImageNet-normalized"

    if workload == "ghnd_bq3":
        from split_heads import GhndBqHead

        head = GhndBqHead(str(CKPT_DIR / GHND_CKPT.format(ch=3)), device)
        frame = _normalized_frame(torch, height, width, device)

        def step():
            payload_bytes, _ = head.encode(frame)
            return payload_bytes

        return step, f"1x3x{height}x{width} uniform random, seed 0, ImageNet-normalized"

    if workload == "fcm_head":
        from split_heads import FcmMaskRcnnHead

        head = FcmMaskRcnnHead(device, min_size=384, max_size=512)
        torch.manual_seed(0)
        frame = torch.rand(3, height, width, device=device)

        def step():
            payload_bytes, _ = head.encode(frame)
            return payload_bytes

        return step, f"3x{height}x{width} uniform random in [0,1], seed 0"

    raise SystemExit(f"unknown workload {workload!r}; choose from {sorted(WORKLOADS)}")


def main():
    workload, mode_label, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_dir = Path(sys.argv[4] if len(sys.argv) > 4 else "results_catalog")
    resolution, description = WORKLOADS[workload]

    import torch

    device = torch.device("cuda")

    idle_sampler = Sampler()
    idle_sampler.start()
    time.sleep(IDLE_S)
    idle_sampler.halt()
    idle_power = [s["total_w"] for s in idle_sampler.samples]

    step, input_description = build_workload(workload, device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()

        run_sampler = Sampler()
        run_sampler.start()
        latencies = []
        payload_bytes = []
        wall_start = time.monotonic()
        while len(latencies) < MIN_ITERS or (time.monotonic() - wall_start) < MIN_WALL_S:
            t0 = time.monotonic()
            payload = step()
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - t0)
            if payload is not None:
                payload_bytes.append(payload)
        wall = time.monotonic() - wall_start
        run_sampler.halt()

    run_power = [s["total_w"] for s in run_sampler.samples]
    idle_mean = statistics.fmean(idle_power)
    run_mean = statistics.fmean(run_power)
    iters = len(latencies)

    rail_names = [r[0] for r in rails()]
    record = {
        "device": device_string(),
        "workload": workload,
        "workload_description": description,
        "mode": mode_label,
        "repeat": repeat,
        "resolution": resolution,
        "input": input_description,
        "iters": iters,
        "wall_s": round(wall, 4),
        "latency_s": dist(latencies),
        "power_idle_w": dist(idle_power),
        "power_run_w": dist(run_power),
        "energy_total_j_per_inf": run_mean * wall / iters,
        "energy_marginal_j_per_inf": max(0.0, run_mean - idle_mean) * wall / iters,
        "payload_bytes": dist(payload_bytes) if payload_bytes else None,
        "rails": rail_names,
        "provenance": (
            "measured: INA3221 module rails ({}) at ~20 Hz; energy = mean power x wall / "
            "iterations; marginal subtracts same-invocation idle; default DVFS governor "
            "(jetson_clocks NOT applied); split-head timed region = preprocessed-tensor "
            "encode (encoder forward + entropy coding / quantization producing real wire "
            "bytes); synthetic uniform-random input (content-independent for semantic "
            "models and heads; Mask R-CNN detection counts on noise are near zero)".format(
                "+".join(rail_names)
            )
        ),
        "raw_idle_samples": idle_sampler.samples,
        "raw_run_samples": run_sampler.samples,
        "raw_latencies_s": [round(v, 5) for v in latencies],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mode_label}__{workload}__rep{repeat}.json"
    path.write_text(json.dumps(record, indent=1))
    payload_note = (
        f" payload={statistics.fmean(payload_bytes) / 1e3:.1f}KB" if payload_bytes else ""
    )
    print(
        f"{mode_label} {workload} rep{repeat}: "
        f"lat={record['latency_s']['mean'] * 1000:.1f}ms "
        f"run={run_mean:.2f}W idle={idle_mean:.2f}W "
        f"E_total={record['energy_total_j_per_inf']:.3f}J "
        f"E_marg={record['energy_marginal_j_per_inf']:.3f}J{payload_note}"
    )


if __name__ == "__main__":
    main()
