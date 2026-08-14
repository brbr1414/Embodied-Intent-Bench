#!/usr/bin/env python3
"""Cross-machine debug: per-stage encoder statistics for the ES and GHND heads (CPU)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from hash_heads import seeded_tensor
from split_heads import EntropicStudentHead, GhndBqHead

CKPT_DIR = Path(sys.argv[1]).expanduser()


def stats(name, t):
    t = t.detach().float()
    print(
        f"{name:<14} shape={tuple(t.shape)} sum={float(t.sum()):.6f} "
        f"absmax={float(t.abs().max()):.6f} nan={int(torch.isnan(t).sum())}"
    )


frame = seeded_tensor()
stats("input", frame)

es = EntropicStudentHead(
    str(
        CKPT_DIR
        / "pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta0.64_from_deeplabv3_resnet50.pt"
    ),
    "cpu",
)
x = frame
with torch.no_grad():
    for i, layer in enumerate(es.encoder):
        x = layer(x)
        stats(f"es.enc[{i}]", x)
    medians = es.entropy_bottleneck.quantiles[:, 0, 1]
    stats("es.medians", medians)
    symbols = torch.round(x - medians.view(1, -1, 1, 1))
    stats("es.symbols", symbols)
    print("es.symbol_hist_absmax", int(symbols.abs().max()))

ghnd = GhndBqHead(
    str(CKPT_DIR / "pascal_voc2012-deeplabv3_resnet50-bq3ch_from_deeplabv3_resnet50.pt"), "cpu"
)
x = frame
with torch.no_grad():
    for i, layer in enumerate(ghnd.encoder):
        x = layer(x)
        stats(f"ghnd.enc[{i}]", x)
