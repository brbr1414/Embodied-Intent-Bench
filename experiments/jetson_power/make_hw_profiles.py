"""Derive hardware-grounded model-zoo scenario profiles from the Jetson measurements.

Reads the frozen stress scenario (``demo_img1_model_zoo.json``) and the local
measurement summaries (``results/jetson_power*/jetson_power_summary.json``,
gitignored), and writes one scenario per measured board:

- ``demo_img1_model_zoo_orin.json``   — AGX Orin 64GB at its default MODE_30W
- ``demo_img1_model_zoo_xavier.json`` — AGX Xavier 32GB at its default MODE_30W_ALL

Derivation rules (owner-approved 2026-08-06):

1. Local configs get the MEASURED ABSOLUTE values at each config's own input
   resolution: ``mission_latency_s`` = mean measured latency, ``energy_j_per_call`` =
   **marginal** energy per inference (compute only).
2. The measured idle floor is folded into ``flight_power_w`` (constant drain is where
   an always-on board belongs, per the A4 cadence finding), and
   ``battery_capacity_wh`` is re-sized so the battery constraint can bind at mission
   scale — a designed stress, recorded in provenance, verified against the static
   baselines before freezing.
3. Remote uploads keep the raw-RGB privacy premise but use the measured raw frame
   sizes; ``uplink_energy_j_per_mb`` (60, UAV-radio stress value) stays configured —
   the B4 measurement showed TX power is invisible on the devkit rails.

The original ``demo_img1_model_zoo.json`` and its documented results are untouched.

    python -m experiments.jetson_power.make_hw_profiles [--battery-wh 3.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path("data/v2_scenarios/demo_img1_model_zoo.json")

#: config_id -> (summary key, resolution key) per board summary.
CONFIG_CELLS = {
    "local_light_real": ("lraspp_mobilenet_v3_large", "512x384"),
    "local_mid_real": ("deeplabv3_mobilenet_v3_large", "512x384"),
    "local_fcn50_real": ("fcn_resnet50", "768x576"),
    "local_strong_real": ("deeplabv3_resnet50", "768x576"),
    "local_heavy_real": ("deeplabv3_resnet101", "768x576"),
}

#: measured raw uint8 RGB frame sizes (content-independent), MB.
RAW_FRAME_MB = {"512x384": 0.5898, "768x576": 1.3271}

BOARDS = {
    "orin": {
        "summary": Path("results/jetson_power/jetson_power_summary.json"),
        "mode": "MODE_30W",
        "device_label": "Jetson AGX Orin 64GB, default MODE_30W, default DVFS",
        "idle_floor_w": 6.7,
        "measured_on": "2026-08-03/05",
    },
    "xavier": {
        "summary": Path("results/jetson_power_xavier/jetson_power_summary.json"),
        "mode": "MODE_30W_ALL",
        "device_label": "Jetson AGX Xavier 32GB, default MODE_30W_ALL, default DVFS",
        "idle_floor_w": 5.9,
        "measured_on": "2026-08-05/06",
    },
}

FLIGHT_POWER_BASE_W = 120.0

#: Periodic ground-station status reports (position, battery, detection summary),
#: sent regardless of where inference runs. Size grounded on the measured protocol
#: envelope (353 B json round-trip, B3) with ~3x headroom for status content;
#: transmit energy reuses the configured 60 J/MB radio stress value (B4: not
#: measurable on the devkit rails).
TELEMETRY = {"interval_s": 1.0, "report_mb": 0.001, "energy_j_per_mb": 60.0, "max_loss_frac": 0.5}


def build_profile(board: str, spec: dict, battery_wh: float) -> Path:
    scenario = json.loads(BASE.read_text())
    cells = json.loads(spec["summary"].read_text())["cells"]

    grounded = {}
    for config in scenario["executor_configs"]:
        cid = config["config_id"]
        if cid in CONFIG_CELLS:
            model, resolution = CONFIG_CELLS[cid]
            cell = cells[f"{spec['mode']}/{model}/{resolution}"]
            config["mission_latency_s"] = round(cell["latency_ms"]["mean"] / 1000.0, 3)
            config["energy_j_per_call"] = round(cell["energy_marginal_j"]["mean"], 3)
            grounded[cid] = {
                "mission_latency_s": config["mission_latency_s"],
                "energy_j_per_call": config["energy_j_per_call"],
                "source_cell": f"{spec['mode']}/{model}/{resolution}",
            }
        else:  # remote tiers: ground the raw-RGB upload size only
            params = config["parameters"]
            resolution = f"{params['backend_input_width_px']}x{params['backend_input_height_px']}"
            config["communication_mb_per_call"] = RAW_FRAME_MB[resolution]
            grounded[cid] = {
                "communication_mb_per_call": config["communication_mb_per_call"],
                "note": "measured raw uint8 RGB size; other remote terms stay configured",
            }

    scenario["scenario_id"] = f"demo_img1_model_zoo_{board}"
    scenario["simulation"]["telemetry"] = dict(TELEMETRY)
    scenario["drone"]["battery_capacity_wh"] = battery_wh
    scenario["drone"]["flight_power_w"] = round(FLIGHT_POWER_BASE_W + spec["idle_floor_w"], 1)
    scenario["mission_contract"]["contract_id"] = f"V2_MODEL_ZOO_CONTRACT_{board.upper()}"

    scenario["provenance"]["design"] = (
        f"hardware-grounded variant of demo_img1_model_zoo: local mission_latency_s and "
        f"energy_j_per_call are MEASURED ABSOLUTE values ({spec['device_label']}; INA3221 "
        f"module rails, marginal energy = run minus same-invocation idle, 3 repeats/cell, "
        f"measured {spec['measured_on']}, experiments/jetson_power/RESULTS.md); the measured "
        f"idle floor ({spec['idle_floor_w']} W) is folded into flight_power_w; "
        f"battery_capacity_wh is a DESIGNED mission-scale stress (not a measurement) sized "
        f"so the battery constraint can bind; remote uploads carry measured raw-RGB frame "
        f"sizes under the unchanged raw-RGB privacy premise, while uplink_energy_j_per_mb, "
        f"remote_compute_s and server energy stay configured stress values (TX power is not "
        f"measurable on the devkit rails — see B4)."
    )
    scenario["provenance"]["hardware_grounding"] = {
        "board": spec["device_label"],
        "measured_on": spec["measured_on"],
        "grounded_values": grounded,
        "telemetry": {
            "report_mb_basis": "measured 353 B protocol envelope (B3) x ~3 headroom",
            "energy_j_per_mb": "configured radio stress value (B4: not measurable)",
        },
        "flight_power_w": {
            "base_flight_fiction_w": FLIGHT_POWER_BASE_W,
            "measured_idle_floor_w": spec["idle_floor_w"],
        },
        "still_configured": [
            "uplink_energy_j_per_mb",
            "downlink_energy_j_per_mb",
            "radio_activation_j",
            "remote_compute_s",
            "remote energy_j_per_call",
            "download_mb_per_call",
            "battery_capacity_wh (designed stress)",
            "flight base power 120 W (flight fiction)",
        ],
    }
    scenario["provenance"]["note"] = (
        "local latency/energy are bench measurements of inference on a devkit (no flight "
        "load, no radio, lab thermals); results are still mission-simulation diagnostics, "
        "never real-UAV performance"
    )

    out = BASE.parent / f"demo_img1_model_zoo_{board}.json"
    out.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-wh", type=float, default=3.0)
    args = parser.parse_args()
    for board, spec in BOARDS.items():
        path = build_profile(board, spec, args.battery_wh)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
