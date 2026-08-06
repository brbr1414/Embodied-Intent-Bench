#!/usr/bin/env python3
"""Client-side remote-path costs on Jetson: encode, decode, protocol, preprocessing.

Grounds the quantities the benchmark's remote model treats as configured or free
(B1/B2/B3/A5 in MEASUREMENT_PLAN.md), on the device that would actually pay them:

- B1  t_encode + real Size_up: JPEG q85 / JPEG q95 / PNG / raw bytes for an RGB frame
      at 512x384 and 768x576.
- B2  t_decode + Size_down: a binary prediction mask serialized as dense JSON
      (the P4 wire format), run-length JSON, and 8-bit PNG; gzip sizes for the JSON
      forms; encode and decode timed separately.
- B3  protocol envelopes: a request dict matching ``InferenceRequest.to_wire()``
      (protocol_version 1.0) and a representative response envelope, json round-trip.
- A5  preprocessing: 1920x1080 capture -> model input (PIL bilinear resize, to-tensor
      + scale, ImageNet normalize, host-to-device copy), per stage.

Content honesty: frames and masks are SYNTHETIC (seeded structured pattern and, for
the encode upper bound, uniform noise). JPEG/PNG sizes are content-dependent; the
structured frame stands in for compressible imagery, the noise frame bounds the
worst case. Neither is real aerial footage and the sizes must be labelled as
synthetic-content measurements. Latency only — no power sampling.

Usage: measure_client.py [out_dir]
"""

import gzip
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

RESOLUTIONS = [(512, 384), (768, 576)]
CAPTURE_WH = (1920, 1080)
PROTOCOL_VERSION = "1.0"


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


def timeit(fn, min_reps=10, min_s=0.3, max_reps=200):
    times = []
    while (len(times) < min_reps or sum(times) < min_s) and len(times) < max_reps:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return dist(times)


def structured_frame(width, height, seed):
    """Compressible synthetic content: low-frequency field + blocks + sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = 120 + 60 * np.sin(2 * np.pi * xx / width * 3.1) * np.cos(2 * np.pi * yy / height * 2.3)
    img = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=-1)
    for _ in range(40):
        x0 = int(rng.integers(0, width - 60))
        y0 = int(rng.integers(0, height - 60))
        img[y0 : y0 + int(rng.integers(8, 60)), x0 : x0 + int(rng.integers(8, 60))] = rng.integers(
            40, 220, size=3
        )
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def noise_frame(width, height, seed):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def blob_mask(width, height, blobs, r_lo, r_hi, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(blobs):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        r = int(rng.integers(r_lo, r_hi))
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    return mask


def measure_frame_codecs(frames):
    """B1: encode/decode time and size per (resolution, content, format)."""
    out = []
    for label, frame in frames:
        h, w = frame.shape[:2]
        image = Image.fromarray(frame)
        cases = [
            ("jpeg_q85", lambda im=image: encode_pil(im, "JPEG", quality=85)),
            ("jpeg_q95", lambda im=image: encode_pil(im, "JPEG", quality=95)),
            ("png", lambda im=image: encode_pil(im, "PNG")),
            ("raw_bytes", lambda fr=frame: fr.tobytes()),
        ]
        for fmt, encoder in cases:
            payload = encoder()
            entry = {
                "content": label,
                "resolution": f"{w}x{h}",
                "format": fmt,
                "size_bytes": len(payload),
                "size_mb": round(len(payload) / 1e6, 4),
                "encode_s": timeit(encoder),
            }
            if fmt != "raw_bytes":
                entry["decode_s"] = timeit(
                    lambda p=payload: np.asarray(Image.open(io.BytesIO(p)).convert("RGB"))
                )
            out.append(entry)
            print(
                f"B1 {label} {w}x{h} {fmt}: {len(payload) / 1e6:.4f} MB "
                f"encode={entry['encode_s']['mean'] * 1000:.2f}ms"
            )
    return out


def encode_pil(image, fmt, **kwargs):
    buf = io.BytesIO()
    image.save(buf, fmt, **kwargs)
    return buf.getvalue()


def rle_encode(mask):
    flat = mask.ravel()
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], boundaries, [flat.size]]))
    return json.dumps({"start_value": int(flat[0]), "runs": runs.tolist()})


def rle_decode(payload, height, width):
    doc = json.loads(payload)
    runs = doc["runs"]
    values = (np.arange(len(runs)) + doc["start_value"]) % 2
    return np.repeat(values.astype(np.uint8), runs).reshape(height, width)


def measure_mask_codecs():
    """B2: dense-JSON vs RLE-JSON vs PNG for sparse and moderate prediction masks."""
    out = []
    for width, height in RESOLUTIONS:
        for density_label, mask in [
            ("sparse", blob_mask(width, height, 3, 3, 9, seed=7)),
            ("moderate", blob_mask(width, height, 8, 15, 40, seed=8)),
        ]:
            positive_frac = float(mask.mean())
            dense = json.dumps(mask.tolist())
            rle = rle_encode(mask)
            png = encode_pil(Image.fromarray(mask * 255), "PNG")
            cases = [
                (
                    "dense_json",
                    dense.encode(),
                    lambda m=mask: json.dumps(m.tolist()).encode(),
                    lambda p=dense: np.array(json.loads(p), dtype=np.uint8),
                    True,
                ),
                (
                    "rle_json",
                    rle.encode(),
                    lambda m=mask: rle_encode(m).encode(),
                    lambda p=rle, hh=height, ww=width: rle_decode(p, hh, ww),
                    True,
                ),
                (
                    "png_mask",
                    png,
                    lambda m=mask: encode_pil(Image.fromarray(m * 255), "PNG"),
                    lambda p=png: np.asarray(Image.open(io.BytesIO(p))),
                    False,
                ),
            ]
            for fmt, payload, encoder, decoder, gzippable in cases:
                entry = {
                    "resolution": f"{width}x{height}",
                    "density": density_label,
                    "positive_frac": round(positive_frac, 5),
                    "format": fmt,
                    "size_bytes": len(payload),
                    "size_mb": round(len(payload) / 1e6, 4),
                    "encode_s": timeit(encoder, min_reps=5),
                    "decode_s": timeit(decoder, min_reps=5),
                }
                if gzippable:
                    zipped = gzip.compress(payload)
                    entry["gzip_size_bytes"] = len(zipped)
                    entry["gzip_decode_s"] = timeit(
                        lambda z=zipped: json.loads(gzip.decompress(z)), min_reps=5
                    )
                out.append(entry)
                print(
                    f"B2 {width}x{height} {density_label} {fmt}: {len(payload) / 1e6:.4f} MB "
                    f"enc={entry['encode_s']['mean'] * 1000:.1f}ms "
                    f"dec={entry['decode_s']['mean'] * 1000:.1f}ms"
                )
    return out


def measure_protocol():
    """B3: json round-trip of the protocol 1.0 request and a representative response."""
    request = {
        "request_id": "scenario_x/000123",
        "scenario_id": "scenario_x",
        "observation_id": 123,
        "capture_time_s": 456.0,
        "deadline_time_s": 900.0,
        "requested_config_id": "CFG_REMOTE_STRONG",
        "payload_kind": "raw_rgb",
        "payload_shape": [384, 512, 3],
        "payload_mb": 0.589824,
        "privacy_level": "none",
        "protocol_version": PROTOCOL_VERSION,
        "payload_ref": "frames/000123.jpg",
    }
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "scenario_x/000123",
        "status": "ok",
        "model_strategy_id": "REMOTE(torch_semantic_segmentation:deeplabv3_resnet101)",
        "stages_s": {
            "request_encoding_s": 0.01,
            "upload_s": 1.2,
            "rtt_s": 0.08,
            "remote_compute_s": 0.05,
            "download_s": 0.3,
            "response_decoding_s": 0.01,
        },
        "mask_ref": "masks/000123.rle",
    }
    out = {}
    for name, doc in [("request", request), ("response_envelope", response)]:
        encoded = json.dumps(doc)
        out[name] = {
            "size_bytes": len(encoded.encode()),
            "encode_s": timeit(lambda d=doc: json.dumps(d), min_s=0.1),
            "decode_s": timeit(lambda e=encoded: json.loads(e), min_s=0.1),
        }
        roundtrip_us = (out[name]["encode_s"]["mean"] + out[name]["decode_s"]["mean"]) * 1e6
        print(f"B3 {name}: {out[name]['size_bytes']} B roundtrip={roundtrip_us:.1f}us")
    return out


def measure_preprocessing():
    """A5: capture-resolution frame -> normalized device tensor, per stage."""
    import torch

    capture = structured_frame(*CAPTURE_WH, seed=11)
    capture_image = Image.fromarray(capture)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for width, height in RESOLUTIONS:
        resized = np.asarray(capture_image.resize((width, height), Image.BILINEAR))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        normalized = (tensor - mean) / std

        def to_device(t=normalized):
            on_gpu = t.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            return on_gpu

        to_device()  # warm the CUDA context before timing
        entry = {
            "capture": f"{CAPTURE_WH[0]}x{CAPTURE_WH[1]} synthetic structured uint8",
            "target": f"{width}x{height}",
            "resize_s": timeit(
                lambda w=width, h=height: capture_image.resize((w, h), Image.BILINEAR)
            ),
            "to_tensor_scale_s": timeit(
                lambda r=resized: torch.from_numpy(r).permute(2, 0, 1).float().div(255.0)
            ),
            "normalize_s": timeit(lambda t=tensor: (t - mean) / std),
            "host_to_device_s": timeit(to_device),
        }
        out.append(entry)
        total_ms = (
            sum(
                entry[stage]["mean"]
                for stage in ("resize_s", "to_tensor_scale_s", "normalize_s", "host_to_device_s")
            )
            * 1000
        )
        print(
            f"A5 -> {width}x{height}: resize={entry['resize_s']['mean'] * 1000:.2f}ms "
            f"tensor={entry['to_tensor_scale_s']['mean'] * 1000:.2f}ms "
            f"norm={entry['normalize_s']['mean'] * 1000:.2f}ms "
            f"h2d={entry['host_to_device_s']['mean'] * 1000:.2f}ms total={total_ms:.2f}ms"
        )
    return out


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    frames = []
    for width, height in RESOLUTIONS:
        frames.append(("structured", structured_frame(width, height, seed=3)))
        frames.append(("noise", noise_frame(width, height, seed=4)))
    record = {
        "kind": "client_side",
        "device": "Jetson AGX Orin 64GB (JetPack R36.4)",
        "b1_frame_codecs": measure_frame_codecs(frames),
        "b2_mask_codecs": measure_mask_codecs(),
        "b3_protocol": measure_protocol(),
        "a5_preprocessing": measure_preprocessing(),
        "provenance": (
            "measured: CPU-side codec/serialization latency and payload sizes on synthetic "
            "content (seeded structured pattern; uniform noise as the incompressible upper "
            "bound) — NOT real aerial imagery; JPEG/PNG sizes are content-dependent. Request "
            "matches InferenceRequest.to_wire() protocol 1.0; the response envelope is "
            "representative only (V3 P1 has no frozen response wire schema). Pillow software "
            "codecs; NVJPEG not used. No power sampling."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "client_side.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out_dir / 'client_side.json'}")


if __name__ == "__main__":
    main()
