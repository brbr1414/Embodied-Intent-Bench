"""Derive a hardware-grounded model-catalog scenario from the Xavier catalog sweep.

Reads the frozen catalog scenario (``demo_img1_model_catalog.json``) and the
catalog-sweep aggregate (``results/jetson_power_xavier/jetson_catalog_summary.json``,
gitignored; 9 workloads x 8 power modes x 3 reps, see RESULTS.md "Xavier catalog
campaign") and writes ``demo_img1_model_catalog_xavier.json`` grounded at one power
mode (default: the board's default MODE_30W_ALL).

Derivation rules (same owner-approved scheme as make_hw_profiles.py, 2026-08-06):

1. Onboard configs (``torch_semantic_segmentation`` / ``torch_instance_segmentation``)
   get MEASURED ABSOLUTE values: ``mission_latency_s`` = mean measured latency at the
   config's own resolution and dtype, ``energy_j_per_call`` = MARGINAL energy per
   inference (compute only).
2. ``pretrained_split`` configs get their onboard-head costs measured from the
   transcribed heads (byte-identical to the real backends, verify_heads.py):
   ``head_latency_s`` = mean head encode latency, ``energy_j_per_call`` = marginal
   head energy. ``communication_mb_per_call`` is NOT overwritten: the committed
   values were measured on this world's mission frames, while the sweep ran
   content-independent random input (entropy-coded ES payloads differ by content:
   21.5 KB random vs 12.6 KB mission frames at beta 0.64 — recorded in provenance).
   ``mission_latency_s`` on split rows stays the configured policy-visible
   expectation; the executed latency is derived per attempt from the network model.
3. The measured idle floor (mode mean across cells) is folded into
   ``flight_power_w``; the raw-RGB remote row keeps its measured upload size and its
   configured server-side terms (TX energy is not measurable on the devkit rails, B4).

MODE_30W_ALL carries DVFS governor jitter on the light workloads (A7 mechanism);
per-cell rep-std is recorded in provenance so the wobble is visible, not laundered.

The original ``demo_img1_model_catalog.json`` and its documented results are
untouched.

    python -m experiments.jetson_power.make_catalog_profile [--mode MODE_30W_ALL]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path("data/v2_scenarios/demo_img1_model_catalog.json")
SUMMARY = Path("results/jetson_power_xavier/jetson_catalog_summary.json")

#: config_id -> sweep workload. Onboard rows ground (mission_latency_s,
#: energy_j_per_call); pretrained_split rows ground (head_latency_s, energy_j_per_call).
ONBOARD_CELLS = {
    "local_light_fp32": "lraspp_fp32",
    "local_light_fp16": "lraspp_fp16",
    "local_strong_fp32": "dlv3_fp32",
    "local_strong_fp16": "dlv3_fp16",
    "maskrcnn_onboard_full": "maskrcnn",
}
HEAD_CELLS = {
    "presplit_es_b064": "es_b064",
    "presplit_es_b512": "es_b512",
    "presplit_ghnd_bq3": "ghnd_bq3",
    "presplit_maskrcnn_fcm": "fcm_head",
}

FLIGHT_POWER_BASE_W = 120.0

#: Resolution-tier rows (--extended): the input-resolution knob on the SAME
#: checkpoints — the strong family at reduced 512x384, the light family at
#: increased 768x576. This is the standard video-analytics configuration axis
#: (resolution trades pixels-on-target against latency/energy) applied to the
#: existing onboard models; no new checkpoints, no quantization search. int8
#: stays out of scope (TensorRT). Each row = (new config_id, base config_id to
#: clone, input WxH, sweep workload for grounding, quality_tier).
RESOLUTION_TIERS = [
    ("local_strong_fp32_512", "local_strong_fp32", (512, 384), "dlv3_fp32_512", "medium"),
    ("local_strong_fp16_512", "local_strong_fp16", (512, 384), "dlv3_fp16_512", "medium"),
    ("local_light_fp32_768", "local_light_fp32", (768, 576), "lraspp_fp32_768", "medium"),
    ("local_light_fp16_768", "local_light_fp16", (768, 576), "lraspp_fp16_768", "medium"),
]


#: Published-quantization rows (--extended): the Qualcomm AI Hub release of
#: DeepLabV3+-MobileNet ships PRE-QUANTIZED w8a8 next to float (VOC2012, 21 classes,
#: person index 15 per the model card; ONNX, runnable on CPU via onnxruntime). Like
#: the pretrained splits, the quantization is ADOPTED as a published artifact —
#: cited, never searched here. Latency/energy are configured stand-ins (Mac CPU
#: wall-clock 146/136 ms w8a8/float) PENDING a board measurement; the local cache
#: path is user-local and gitignored.
QAIHUB_PROVENANCE = (
    "Qualcomm AI Hub 'DeepLabV3-Plus-MobileNet' release v0.60.0 "
    "(huggingface.co/qualcomm/DeepLabV3-Plus-MobileNet; Qualcomm AI Hub license); "
    "VOC2012, 21 classes, person index 15 from the model card; ONNX asset run "
    "unmodified via onnxruntime CPU EP at ORT_ENABLE_BASIC"
)
ONNX_ROWS = [
    {
        "config_id": "local_dlv3plus_w8a8",
        "model_strategy_id": "QAIHUB_DLV3PLUS_MOBILENET_W8A8",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.146,
        "energy_j_per_call": 1.0,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/deeplabv3_plus_mobilenet-onnx-w8a8/"
                "deeplabv3_plus_mobilenet-onnx-w8a8/deeplabv3_plus_mobilenet.onnx"
            ),
            "person_index": 15,
            "input_width_px": 520,
            "input_height_px": 520,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": QAIHUB_PROVENANCE + "; precision w8a8 (published)",
        },
    },
    {
        "config_id": "local_dlv3plus_fp32_onnx",
        "model_strategy_id": "QAIHUB_DLV3PLUS_MOBILENET_FLOAT",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.136,
        "energy_j_per_call": 1.2,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/deeplabv3_plus_mobilenet-onnx-float/"
                "deeplabv3_plus_mobilenet-onnx-float/deeplabv3_plus_mobilenet.onnx"
            ),
            "person_index": 15,
            "input_width_px": 520,
            "input_height_px": 520,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": QAIHUB_PROVENANCE + "; precision float (paired ablation row)",
        },
    },
]

ONNX_ROWS += [
    {
        "config_id": "local_segformer_w8a8",
        "model_strategy_id": "QAIHUB_SEGFORMER_B0_ADE_W8A8",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.098,
        "energy_j_per_call": 0.8,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/segformer_base-onnx-w8a8/"
                "segformer_base-onnx-w8a8/segformer_base.onnx"
            ),
            "person_index": 12,
            "input_width_px": 512,
            "input_height_px": 512,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": (
                "Qualcomm AI Hub 'segformer_base' release v0.60.0 (huggingface.co/qualcomm; "
                "Qualcomm AI Hub license); SegFormer-B0 fine-tuned on ADE20K, 150 classes, "
                "person index 12 (3.75M params); ONNX w8a8 run unmodified via "
                "onnxruntime CPU EP at ORT_ENABLE_BASIC"
            ),
        },
    },
    {
        "config_id": "local_segformer_fp32_onnx",
        "model_strategy_id": "QAIHUB_SEGFORMER_B0_ADE_FLOAT",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.069,
        "energy_j_per_call": 0.9,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/segformer_base-onnx-float/"
                "segformer_base-onnx-float/segformer_base.onnx"
            ),
            "person_index": 12,
            "input_width_px": 512,
            "input_height_px": 512,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": (
                "Qualcomm AI Hub 'segformer_base' release v0.60.0 (huggingface.co/qualcomm; "
                "Qualcomm AI Hub license); SegFormer-B0 fine-tuned on ADE20K, 150 classes, "
                "person index 12 (3.75M params); ONNX float run unmodified via "
                "onnxruntime CPU EP at ORT_ENABLE_BASIC"
            ),
        },
    },
    {
        "config_id": "local_ffnet40s_w8a8",
        "model_strategy_id": "QAIHUB_FFNET_40S_CITYSCAPES_W8A8",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.318,
        "energy_j_per_call": 3.0,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/ffnet_40s-onnx-w8a8/ffnet_40s-onnx-w8a8/ffnet_40s.onnx"
            ),
            "person_index": 11,
            "input_width_px": 2048,
            "input_height_px": 1024,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": (
                "Qualcomm AI Hub 'ffnet_40s' release v0.60.0 (huggingface.co/qualcomm; "
                "Qualcomm AI Hub license); FFNet-40S on Cityscapes, 19 train classes, "
                "person index 11 (13.9M params); ONNX w8a8 run unmodified via "
                "onnxruntime CPU EP at ORT_ENABLE_BASIC; w8a8 measured FASTER "
                "than float on Mac CPU (318 vs 408 ms)"
            ),
        },
    },
    {
        "config_id": "local_ffnet40s_fp32_onnx",
        "model_strategy_id": "QAIHUB_FFNET_40S_CITYSCAPES_FLOAT",
        "kind": "onnx_semantic_segmentation",
        "mission_latency_s": 0.408,
        "energy_j_per_call": 3.6,
        "communication_mb_per_call": 0.0,
        "quality_tier": "medium",
        "parameters": {
            "model_path": (
                "~/.cache/qaihub/ffnet_40s-onnx-float/ffnet_40s-onnx-float/ffnet_40s.onnx"
            ),
            "person_index": 11,
            "input_width_px": 2048,
            "input_height_px": 1024,
            "latency_mode": "configured",
            "min_component_px": 0,
            "quantization_source": (
                "Qualcomm AI Hub 'ffnet_40s' release v0.60.0 (huggingface.co/qualcomm; "
                "Qualcomm AI Hub license); FFNet-40S on Cityscapes, 19 train classes, "
                "person index 11 (13.9M params); ONNX float run unmodified via "
                "onnxruntime CPU EP at ORT_ENABLE_BASIC"
            ),
        },
    },
]


def _resolution_tier_rows(configs: list[dict], cells: dict, mode: str) -> list[dict]:
    by_id = {config["config_id"]: config for config in configs}
    rows = []
    for new_id, base_id, (width, height), workload, tier in RESOLUTION_TIERS:
        cell = cells[f"{mode}/{workload}"]
        row = json.loads(json.dumps(by_id[base_id]))  # deep copy
        row["config_id"] = new_id
        row["model_strategy_id"] = by_id[base_id]["model_strategy_id"] + f"_{width}"
        row["quality_tier"] = tier
        row["parameters"]["input_width_px"] = width
        row["parameters"]["input_height_px"] = height
        row["mission_latency_s"] = round(cell["latency_ms"]["mean"] / 1000.0, 3)
        row["energy_j_per_call"] = round(cell["energy_marginal_j"]["mean"], 3)
        rows.append(row)
    return rows


def build_profile(mode: str, extended: bool = False) -> Path:
    scenario = json.loads(BASE.read_text())
    summary = json.loads(SUMMARY.read_text())
    cells = summary["cells"]

    def cell(workload: str) -> dict:
        return cells[f"{mode}/{workload}"]

    grounded: dict[str, dict] = {}
    for config in scenario["executor_configs"]:
        cid = config["config_id"]
        if cid in ONBOARD_CELLS:
            c = cell(ONBOARD_CELLS[cid])
            config["mission_latency_s"] = round(c["latency_ms"]["mean"] / 1000.0, 3)
            config["energy_j_per_call"] = round(c["energy_marginal_j"]["mean"], 3)
            grounded[cid] = {
                "mission_latency_s": config["mission_latency_s"],
                "latency_rep_std_ms": round(c["latency_ms"]["std"], 1),
                "energy_j_per_call": config["energy_j_per_call"],
                "source_cell": f"{mode}/{ONBOARD_CELLS[cid]}",
            }
        elif cid in HEAD_CELLS:
            c = cell(HEAD_CELLS[cid])
            config["parameters"]["head_latency_s"] = round(c["latency_ms"]["mean"] / 1000.0, 3)
            config["energy_j_per_call"] = round(c["energy_marginal_j"]["mean"], 3)
            grounded[cid] = {
                "head_latency_s": config["parameters"]["head_latency_s"],
                "latency_rep_std_ms": round(c["latency_ms"]["std"], 1),
                "energy_j_per_call": config["energy_j_per_call"],
                "payload_note": (
                    "communication_mb_per_call kept at the mission-frame measurement; "
                    f"sweep random-input payload was {c['payload_kb']['mean']:.1f} KB"
                ),
                "source_cell": f"{mode}/{HEAD_CELLS[cid]}",
            }
        # remote_strong_raw: measured raw-RGB upload size already committed; server
        # side stays configured.

    workloads = set(ONBOARD_CELLS.values()) | set(HEAD_CELLS.values())
    idle_w = round(
        sum(cells[f"{mode}/{w}"]["power_idle_w"]["mean"] for w in workloads) / len(workloads),
        1,
    )

    if extended:
        extra = _resolution_tier_rows(scenario["executor_configs"], cells, mode)
        scenario["executor_configs"].extend(extra)
        for row in extra:
            grounded[row["config_id"]] = {
                "mission_latency_s": row["mission_latency_s"],
                "energy_j_per_call": row["energy_j_per_call"],
                "source_cell": f"{mode}/(resolution-tier sweep)",
            }
        scenario["executor_configs"].extend(json.loads(json.dumps(ONNX_ROWS)))
        for row in ONNX_ROWS:
            grounded[row["config_id"]] = {
                "mission_latency_s": row["mission_latency_s"],
                "energy_j_per_call": row["energy_j_per_call"],
                "source_cell": (
                    "CONFIGURED STAND-IN (Mac CPU wall-clock; energy designed) — "
                    "PENDING board measurement"
                ),
            }
    suffix = "xavier" if mode == "MODE_30W_ALL" else f"xavier_{mode.lower()}"
    if extended:
        suffix += "_ext"
    scenario["scenario_id"] = f"demo_img1_model_catalog_{suffix}"
    scenario["drone"]["flight_power_w"] = round(FLIGHT_POWER_BASE_W + idle_w, 1)
    scenario["mission_contract"]["contract_id"] = f"V2_MODEL_CATALOG_CONTRACT_{suffix.upper()}"

    scenario["provenance"]["design"] = (
        scenario["provenance"]["design"]
        + f" HARDWARE-GROUNDED VARIANT ({mode}): onboard mission_latency_s / energy_j_per_call"
        " and split-head head_latency_s / energy_j_per_call are MEASURED ABSOLUTE values"
        " (Jetson AGX Xavier 32GB, INA3221 module rails, marginal energy = run minus"
        " same-invocation idle, 3 repeats/cell, 2026-08-13/14 catalog campaign,"
        " experiments/jetson_power/RESULTS.md); heads are transcriptions pinned"
        " byte-identical to the real backends (verify_heads.py). The measured idle floor"
        f" ({idle_w} W mode mean) is folded into flight_power_w. Split payload sizes keep"
        " the mission-frame measurements (sweep random-input payloads recorded in"
        " hardware_grounding). Server-side and radio terms stay configured."
    )
    scenario["provenance"]["hardware_grounding"] = {
        "board": f"Jetson AGX Xavier 32GB, {mode}, default DVFS governor",
        "measured_on": "2026-08-13/14",
        "source_summary": str(SUMMARY),
        "grounded_values": grounded,
        "flight_power_w": {
            "base_flight_fiction_w": FLIGHT_POWER_BASE_W,
            "measured_idle_floor_w": idle_w,
        },
        "dvfs_note": (
            "MODE_30W_ALL shows governor-entangled latency jitter on light workloads "
            "(A7); per-cell rep std is recorded above and not smoothed away"
        ),
        "still_configured": [
            "mission_latency_s on pretrained_split rows (policy-visible expectation; "
            "executed latency is derived)",
            "remote_compute_s and server energy (all remote/split rows)",
            "uplink/downlink energy_j_per_mb and radio_activation_j (B4: TX not measurable)",
            "download_mb_per_call",
            "telemetry energy_j_per_mb",
            "battery_capacity_wh (designed mission stress, unchanged from base)",
            "flight base power 120 W (flight fiction)",
        ],
    }
    scenario["provenance"]["note"] = (
        "local/head latency and energy are bench measurements of inference on a devkit "
        "(no flight load, no radio, lab thermals); results are still mission-simulation "
        "diagnostics, never real-UAV performance"
    )

    out = BASE.parent / f"demo_img1_model_catalog_{suffix}.json"
    out.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="MODE_30W_ALL")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="add measured resolution-tier rows (strong@512, light@768) as new configs",
    )
    args = parser.parse_args()
    path = build_profile(args.mode, extended=args.extended)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
