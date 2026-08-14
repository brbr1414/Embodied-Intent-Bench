#!/usr/bin/env python3
"""Verify the transcribed board-measurement heads against the benchmark's real backends.

Runs on the development machine (needs the [v2-presplit] stack: sc2bench + torch).
For each catalog split, the same seeded frame goes through the genuine presplit
backend and the split_heads transcription; the transcription passes only if the
payload it encodes is byte-identical (ES: entropy-coded strings; GHND: quantized
latent tensor + scale/zero-point; FCM: per-tensor wire byte counts). A board
measurement of a head that fails this check would be a measurement of the wrong
computation — run this before trusting any board number.

Usage: ~/.venvs/sc2bench/bin/python verify_heads.py [ckpt_root]
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from split_heads import EntropicStudentHead, FcmMaskRcnnHead, GhndBqHead

from aerointentbench.v2.instance_models import FcmMaskRcnnSplitModel
from aerointentbench.v2.presplit import _Sc2EntropicStudentModel, _Sc2GhndBqModel
from aerointentbench.v2.scenario import ExecutorConfigSpec

CKPT_ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else "~/.cache/sc2-benchmark/resource/ckpt/pascal_voc2012"
).expanduser()
ES_DIR = CKPT_ROOT / "supervised_compression/entropic_student"
GHND_DIR = CKPT_ROOT / "supervised_compression/ghnd-bq"

WIDTH, HEIGHT = 512, 384


def spec_for(config_id, kind, parameters):
    return ExecutorConfigSpec(
        config_id=config_id,
        model_strategy_id=config_id,
        kind=kind,
        mission_latency_s=0.4,
        energy_j_per_call=8.0,
        communication_mb_per_call=0.1,
        quality_tier=1,
        parameters=parameters,
    )


def seeded_rgb():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


def verify_es(beta):
    name = f"pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta{beta}_from_deeplabv3_resnet50.pt"
    ckpt = ES_DIR / name
    real = _Sc2EntropicStudentModel(
        spec_for(
            f"verify_es_{beta}",
            "pretrained_split",
            {
                "split_backend": "sc2_entropic_student",
                "checkpoint_path": str(ckpt),
                "input_width_px": WIDTH,
                "input_height_px": HEIGHT,
                "device": "cpu",
            },
        )
    )
    rgb = seeded_rgb()
    tensor = real._preprocess(rgb)
    with torch.no_grad():
        real_payload = real._model.backbone.bottleneck_layer.encode(tensor)
    real_strings = real_payload["strings"][0]

    head = EntropicStudentHead(str(ckpt), "cpu")
    head_bytes, head_payload = head.encode(tensor)
    head_strings = head_payload["strings"][0]

    assert len(real_strings) == len(head_strings)
    assert all(a == b for a, b in zip(real_strings, head_strings, strict=True)), (
        f"ES {beta}: strings differ"
    )
    real_bytes = sum(len(s) for s in real_strings)
    assert head_bytes == real_bytes, f"ES {beta}: {head_bytes} != {real_bytes}"
    print(f"ES beta{beta}: OK — {head_bytes} payload bytes, byte-identical bitstream")


def verify_ghnd():
    ckpt = GHND_DIR / "pascal_voc2012-deeplabv3_resnet50-bq3ch_from_deeplabv3_resnet50.pt"
    real = _Sc2GhndBqModel(
        spec_for(
            "verify_ghnd_bq3",
            "pretrained_split",
            {
                "split_backend": "sc2_ghnd_bq",
                "checkpoint_path": str(ckpt),
                "input_width_px": WIDTH,
                "input_height_px": HEIGHT,
                "device": "cpu",
                "bottleneck_channels": 3,
            },
        )
    )
    rgb = seeded_rgb()
    tensor = real._preprocess(rgb)
    with torch.no_grad():
        real_payload = real._model.backbone.bottleneck_layer.encode(tensor)
    real_q = real_payload["z"]

    head = GhndBqHead(str(ckpt), "cpu")
    head_bytes, head_q = head.encode(tensor)

    assert torch.equal(real_q.tensor, head_q.tensor), "GHND: quantized latents differ"
    assert float(real_q.scale) == float(head_q.scale), "GHND: scales differ"
    assert int(real_q.zero_point) == int(head_q.zero_point), "GHND: zero points differ"
    assert head_bytes == int(real_q.tensor.numel())
    print(f"GHND bq3: OK — {head_bytes} payload bytes, identical quantized latent")


def verify_fcm():
    real = FcmMaskRcnnSplitModel(
        spec_for(
            "verify_fcm",
            "pretrained_split",
            {
                "split_backend": "fcm_maskrcnn_fpn",
                "score_threshold": 0.5,
                "input_min_size_px": 384,
                "input_max_size_px": 512,
                "device": "cpu",
                "feature_dtype": "uint8",
            },
        )
    )
    rgb = seeded_rgb()
    _, payload_mb, diagnostics = real.encode(rgb)
    real_detail = diagnostics["feature_payload"]

    head = FcmMaskRcnnHead("cpu", min_size=384, max_size=512)
    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    )
    head_bytes, wires = head.encode(tensor)

    real_bytes = sum(entry["wire_bytes"] for entry in real_detail.values())
    assert set(wires) == set(real_detail), "FCM: crossing tensor names differ"
    for name, (wire, _, _) in wires.items():
        assert wire.numel() == real_detail[name]["wire_bytes"], f"FCM {name}: wire bytes differ"
    assert head_bytes == real_bytes == round(payload_mb * 1e6)
    print(f"FCM maskrcnn: OK — {head_bytes} payload bytes across {len(wires)} P-layers")


if __name__ == "__main__":
    verify_es("0.64")
    verify_es("5.12")
    verify_ghnd()
    verify_fcm()
    print("ALL HEADS VERIFIED")
