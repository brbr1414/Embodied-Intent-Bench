"""The V2 CLI: validate a scenario, render an overview, run a visual mission.

    python -m aerointentbench.v2.cli validate --scenario data/v2_scenarios/demo.json
    python -m aerointentbench.v2.cli overview --scenario ... --output results/v2/
    python -m aerointentbench.v2.cli run --scenario ... --policy rule_based --output results/v2/

Separate from the V1 ``run_benchmark`` entry point on purpose: V1's CLI and behaviour
are frozen, and a V2 mission takes a scenario, not an episode/contract pair. The banner
always states the mode and the honesty labels -- visual V2 mode, simulated latency and
energy, synthetic targets -- so a result can never silently masquerade as either V1
replay or real hardware measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aerointentbench.schemas.loading import SchemaValidationError, SchemaVersionError
from aerointentbench.v2.scenario import V2Scenario, load_scenario

__all__ = ["main"]


def _banner(scenario: V2Scenario, policy: str | None) -> str:
    torch_specs = [s for s in scenario.executor_configs if s.kind == "torch_semantic_segmentation"]
    if torch_specs:
        modes = {str(s.parameters["latency_mode"]) for s in torch_specs}
        latency_line = (
            "latency: MEASURED wall-clock drives the mission clock (machine-dependent)"
            if modes == {"measured"}
            else f"latency: per-executor latency_mode {sorted(modes)} "
            "(measured wall-clock always recorded as diagnostic)"
        )
    else:
        latency_line = (
            "latency: SIMULATED (configured per executor; wall-clock recorded as diagnostic)"
        )
    lines = [
        "mode: V2 visual closed loop (V1 profile replay remains available via "
        "aerointentbench.run_benchmark)",
        f"world image: {scenario.world.image_path} (source {scenario.world.source_id})",
        f"executors: {', '.join(f'{s.config_id} [{s.kind}]' for s in scenario.executor_configs)}",
        latency_line,
        "energy/communication: SIMULATED/CONFIGURED (no hardware measurement)",
        "targets: SYNTHETIC rescue markers composited over real aerial imagery",
    ]
    for spec in torch_specs:
        p = spec.parameters
        lines.append(
            f"real model {spec.config_id}: {p['model_id']} (weights {p['weights_id']}, "
            f"class {p['task_class']!r}, input {p['input_width_px']}x{p['input_height_px']}, "
            f"device {p['device']}, dtype {p['dtype']}) -- pretrained COCO/VOC weights; "
            "expect DOMAIN MISMATCH on the synthetic markers"
        )
    if policy is not None:
        lines.insert(2, f"policy: {policy}")
    return "\n".join(lines)


def _cmd_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    print(_banner(scenario, None))
    print(
        f"scenario {scenario.scenario_id!r} is valid: {len(scenario.targets)} targets, "
        f"{len(scenario.distractors)} distractors, {len(scenario.executor_configs)} executors"
    )
    return 0


def _cmd_overview(args: argparse.Namespace) -> int:
    from aerointentbench.v2.objects import ObjectLayer
    from aerointentbench.v2.trajectory import build_trajectory
    from aerointentbench.v2.visualize import render_overview
    from aerointentbench.v2.world import open_world

    scenario = load_scenario(args.scenario)
    print(_banner(scenario, None))
    world = open_world(
        Path(scenario.world.image_path),
        scenario.world.meters_per_pixel,
        invalid_pixel_rule=scenario.world.invalid_pixel_rule,
    )
    path = render_overview(
        scenario,
        world,
        build_trajectory(scenario.trajectory, scenario.drone.speed_mps),
        ObjectLayer(scenario.objects),
        args.output / f"{scenario.scenario_id}_overview.png",
    )
    print(f"wrote {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from aerointentbench.v2.runner import MissionRunner
    from aerointentbench.v2.visualize import render_observation_panel

    scenario = load_scenario(args.scenario)
    print(_banner(scenario, args.policy))

    runner = MissionRunner(scenario, args.policy)
    result = runner.run()

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"{scenario.scenario_id}_{args.policy}.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.debug_observations > 0:
        step = max(1, len(result.observations) // args.debug_observations)
        for log in result.observations[::step][: args.debug_observations]:
            observation = runner.render_at(log.capture_time_s, log.observation_id)
            executor = runner._executors[log.executed_config_id]
            panel = render_observation_panel(
                observation,
                executor.run(observation.rgb).prediction_mask,
                args.output / f"{scenario.scenario_id}_obs{log.observation_id:04d}.png",
                annotation={
                    "config": log.executed_config_id,
                    "t_done": f"{log.completion_time_s:.1f}s",
                    "pos_done": (
                        f"({log.completion_position_m[0]:.1f}, {log.completion_position_m[1]:.1f})m"
                    ),
                },
            )
            print(f"wrote {panel}")

    q = result.quality
    print(
        f"\nmission_success={result.mission_success}  termination={result.termination_reason}\n"
        f"target_recall={q['target_recall']:.3f}  "
        f"detection_precision={q['detection_precision']:.3f}  "
        f"fp={q['false_positive_detections']}\n"
        f"processed={result.processed_observation_count}  "
        f"skipped={result.skipped_observation_count}  "
        f"battery={result.final_battery_frac:.3f}  time={result.final_time_s:.1f}s\n"
        f"wrote {result_path}"
    )
    return 0


def _cmd_check_real_models(args: argparse.Namespace) -> int:
    import json as json_module

    from aerointentbench.v2.real_models import check_real_model_availability

    scenario = load_scenario(args.scenario)
    torch_specs = [s for s in scenario.executor_configs if s.kind == "torch_semantic_segmentation"]
    if not torch_specs:
        print("no torch_semantic_segmentation executors in this scenario")
        return 0
    reports = check_real_model_availability(
        [dict(s.parameters) for s in torch_specs], load=args.load
    )
    print(json_module.dumps(reports, indent=2))
    return 0 if all(r.get("status") == "ok" for r in reports) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aerointentbench.v2.cli", description="AeroIntentBench V2 visual missions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate a V2 scenario file.")
    p_validate.add_argument("--scenario", type=Path, required=True)
    p_validate.set_defaults(fn=_cmd_validate)

    p_check = sub.add_parser(
        "check-real-models",
        help="Verify real-model executors are runnable (imports, weights, target class).",
    )
    p_check.add_argument("--scenario", type=Path, required=True)
    p_check.add_argument(
        "--load",
        action="store_true",
        help="Also construct each backend (downloads/caches official weights).",
    )
    p_check.set_defaults(fn=_cmd_check_real_models)

    p_overview = sub.add_parser("overview", help="Render the world/trajectory overview PNG.")
    p_overview.add_argument("--scenario", type=Path, required=True)
    p_overview.add_argument("--output", type=Path, required=True)
    p_overview.set_defaults(fn=_cmd_overview)

    p_run = sub.add_parser("run", help="Run a V2 visual mission.")
    p_run.add_argument("--scenario", type=Path, required=True)
    p_run.add_argument("--policy", default="rule_based")
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument(
        "--debug-observations",
        type=int,
        default=0,
        metavar="N",
        help="Export up to N observation debug panels (default: none; runs stay headless).",
    )
    p_run.set_defaults(fn=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (SchemaValidationError, SchemaVersionError) as error:
        print(f"aerointentbench.v2: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
