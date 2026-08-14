#!/usr/bin/env python3
"""Print a deterministic payload hash per split head, for cross-machine verification.

The development machine (compressai 1.2.6) and the board (compressai 1.2.8) must
produce the SAME entropy-coded bitstream from the same checkpoint and input —
compress() reads only the checkpoint's CDF tables, which the version change did
not touch. This script builds each head on CPU, feeds the same seeded frame the
Mac verification used, and prints payload bytes + sha256; identical lines on both
machines close the version-skew question. Python 3.8 compatible.

Usage: hash_heads.py <ckpt_dir>
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from split_heads import EntropicStudentHead, GhndBqHead

WIDTH, HEIGHT = 512, 384


def seeded_tensor():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    array = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)).unsqueeze(0)


def main():
    ckpt_dir = Path(sys.argv[1]).expanduser()
    frame = seeded_tensor()
    for beta in ("0.64", "5.12"):
        head = EntropicStudentHead(
            str(
                ckpt_dir
                / (
                    f"pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta{beta}"
                    "_from_deeplabv3_resnet50.pt"
                )
            ),
            "cpu",
        )
        n, payload = head.encode(frame)
        digest = hashlib.sha256(b"".join(payload["strings"][0])).hexdigest()[:16]
        print(f"es_b{beta.replace('.', '')} bytes={n} sha256={digest}")

    head = GhndBqHead(
        str(ckpt_dir / "pascal_voc2012-deeplabv3_resnet50-bq3ch_from_deeplabv3_resnet50.pt"),
        "cpu",
    )
    n, quantized = head.encode(frame)
    digest = hashlib.sha256(quantized.tensor.numpy().tobytes()).hexdigest()[:16]
    print(f"ghnd_bq3 bytes={n} sha256={digest} zero_point={quantized.zero_point}")


if __name__ == "__main__":
    main()
