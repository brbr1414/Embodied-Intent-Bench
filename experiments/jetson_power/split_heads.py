#!/usr/bin/env python3
"""Standalone onboard HEAD modules of the catalog's pre-split models, for board measurement.

The benchmark's genuine presplit backends (aerointentbench.v2.presplit / instance_models)
need the full sc2bench dependency chain, which does not install on the Jetson's
python 3.8 / torch 2.1 stack. This module transcribes ONLY the onboard head of each
published split so the head's latency/energy can be measured on the board:

- ES (Entropic Student, WACV 2022 / SC2 TMLR 2023): FPBasedResNetBottleneck encoder
  (3 convs + 2 GDN1) + compressai EntropyBottleneck.compress — transcribed verbatim
  from sc2bench.models.layer.FPBasedResNetBottleneck. Needs compressai only.
- GHND-BQ (Matsubara et al., IEEE Access 2020): the v0.0.3-era encoder module list
  (already transcribed in aerointentbench.v2.presplit) + Jacob et al. 8-bit
  quantization transcribed from torchdistill.common.tensor_util. Plain torch.
- FCM (MPEG FCM split point via CompressAI-Vision): torchvision Mask R-CNN
  transform + backbone + FPN, with the benchmark's per-tensor uint8 affine pricing
  transcribed from aerointentbench.v2.split.encode_features. Plain torchvision.

Transcription is measurement plumbing, not design: every head must produce
byte-identical payloads to the benchmark's real backend (verified on the
development machine by verify_heads.py before any board number is trusted).
Python 3.8 compatible.
"""

import collections

import torch
from torch import nn

QuantizedTensor = collections.namedtuple("QuantizedTensor", ["tensor", "scale", "zero_point"])


def _strict_subload(module, full_state, prefix, shape_buffers=()):
    """Load `prefix`-keyed entries of a checkpoint into `module`, strictly.

    `shape_buffers` names buffers whose size is checkpoint-defined (compressai CDF
    tables); they are shaped from the checkpoint first so the strict load verifies
    every key.
    """
    sub = {}
    for key, value in full_state.items():
        if key.startswith(prefix):
            sub[key[len(prefix) :]] = value
    own = dict(module.state_dict())
    # compressai >= 1.2.7 renamed EntropyBottleneck's factorized-prior parameters
    # (_matrix0 -> matrices.0, _bias0 -> biases.0, _factor0 -> factors.0). The SC2
    # checkpoints carry the old names; remap when the installed module uses the new
    # ones. compress() itself reads only the CDF buffers, whose names are unchanged.
    renames = (("_matrix", "matrices."), ("_bias", "biases."), ("_factor", "factors."))
    for old_stem, new_stem in renames:
        for key in list(sub):
            head, sep, index = key.rpartition(old_stem)
            if sep and index.isdigit() and key not in own and (head + new_stem + index) in own:
                sub[head + new_stem + index] = sub.pop(key)
    picked = {k: v for k, v in sub.items() if k in own}
    for name in shape_buffers:
        obj = module
        *parents, leaf = name.split(".")
        for part in parents:
            obj = getattr(obj, part)
        setattr(obj, leaf, torch.empty_like(picked[name]))
    missing = set(module.state_dict()) - set(picked)
    if missing:
        raise RuntimeError(f"checkpoint missing keys for head: {sorted(missing)}")
    module.load_state_dict(picked, strict=True)


class EntropicStudentHead(nn.Module):
    """ES onboard head: encoder -> EntropyBottleneck.compress -> bitstream bytes."""

    def __init__(self, checkpoint_path, device, bottleneck_channels=24):
        super().__init__()
        from compressai.entropy_models import EntropyBottleneck
        from compressai.layers import GDN1

        c = bottleneck_channels
        # Verbatim from sc2bench FPBasedResNetBottleneck (encoder_channel_sizes
        # default: [3, 4c, 2c, c]).
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(c * 4),
            nn.Conv2d(c * 4, c * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(c * 2),
            nn.Conv2d(c * 2, c, kernel_size=2, stride=1, padding=0, bias=False),
        )
        self.entropy_bottleneck = EntropyBottleneck(c)
        # Same trust decision as aerointentbench.v2.presplit: SC2 checkpoints carry an
        # argparse.Namespace, so a weights-only load rejects them.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        _strict_subload(
            self,
            state,
            "backbone.bottleneck_layer.",
            shape_buffers=(
                "entropy_bottleneck._offset",
                "entropy_bottleneck._quantized_cdf",
                "entropy_bottleneck._cdf_length",
            ),
        )
        self.to(device).eval()

    def encode(self, tensor):
        """Return (payload_bytes, payload) for one preprocessed frame tensor."""
        with torch.no_grad():
            latent = self.encoder(tensor)
            strings = self.entropy_bottleneck.compress(latent)
        return sum(len(s) for s in strings), {"strings": [strings], "shape": latent.size()[-2:]}


class GhndBqHead(nn.Module):
    """GHND-BQ onboard head: v0.0.3 encoder (modules[:12]) -> 8-bit quantized latent."""

    def __init__(self, checkpoint_path, device, bottleneck_channel=3):
        super().__init__()
        # Verbatim module list from sc2-benchmark v0.0.3 larger_resnet_bottleneck
        # (same transcription as aerointentbench.v2.presplit._Sc2GhndBqModel).
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        _strict_subload(self, state, "backbone.bottleneck_layer.")
        self.to(device).eval()

    @staticmethod
    def _quantize_tensor(x, num_bits=8):
        # Verbatim from torchdistill.common.tensor_util.quantize_tensor (Jacob et al.).
        qmin = 0.0
        qmax = 2.0**num_bits - 1.0
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (qmax - qmin)
        initial_zero_point = qmin - min_val / scale
        zero_point = (
            qmin
            if initial_zero_point < qmin
            else qmax
            if initial_zero_point > qmax
            else initial_zero_point
        )
        zero_point = int(zero_point)
        qx = zero_point + x / scale
        qx = qx.clamp(qmin, qmax).round().byte()
        return QuantizedTensor(tensor=qx, scale=scale, zero_point=zero_point)

    def encode(self, tensor):
        with torch.no_grad():
            z = self.encoder(tensor)
            quantized = self._quantize_tensor(z, 8)
        return int(quantized.tensor.numel()), quantized


class FcmMaskRcnnHead:
    """FCM onboard head: Mask R-CNN transform + backbone + FPN -> uint8-priced features."""

    def __init__(self, device, min_size=384, max_size=512):
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
        self.model.to(device).eval()
        self.device = device

    def encode(self, tensor):
        """tensor: 3xHxW float in [0,1] on device. Returns (payload_bytes, features)."""
        with torch.no_grad():
            images, _ = self.model.transform([tensor], None)
            features = self.model.backbone(images.tensors)
            total_bytes = 0
            wires = {}
            for name, feature in features.items():
                # Per-tensor uint8 affine, verbatim semantics of
                # aerointentbench.v2.split.encode_features (torch instead of numpy;
                # byte count is shape-determined either way).
                source = feature[0]
                low = source.min()
                high = source.max()
                scale = (high - low) / 255.0
                scale = scale if float(scale) != 0.0 else torch.ones_like(scale)
                wire = torch.clamp(torch.round((source - low) / scale), 0, 255).to(torch.uint8)
                total_bytes += wire.numel()
                wires[name] = (wire, scale, low)
        return int(total_bytes), wires
