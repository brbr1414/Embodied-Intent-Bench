"""Run a V2 mission with the LLM policy and write an honest report.

Composes LlmPolicy + TransformersBackend into MissionRunner through the injection seam
(the LLM stack never touches the core package). Decision wall-clock is measured around
each ``select_config`` call and reported as a DIAGNOSTIC — mission time is charged only
by the scenario's ``policy_execution`` block, if the scenario declares one. The report
records the model id, device, mode, per-decision choices, parse failures, and the
mission result, so a number can always be traced to what produced it.

    ~/.venvs/sc2bench/bin/python -m experiments.llm_policy.run_llm_mission \
        --scenario data/v2_scenarios/demo_img1_model_catalog_xavier.json \
        --mode choice --output results/llm_policy/
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from aerointentbench.v2.runner import MissionRunner
from aerointentbench.v2.scenario import load_scenario
from experiments.llm_policy.backend import DEFAULT_MODEL_ID, TransformersBackend
from experiments.llm_policy.policy import LlmPolicy


class _TimedPolicy:
    """Delegates to the LLM policy, measuring wall-clock per decision (diagnostic only)."""

    def __init__(self, inner: LlmPolicy) -> None:
        self._inner = inner
        self.diagnostics = inner.diagnostics

    def select_config(self, contract, state, configs):
        started = time.perf_counter()
        try:
            return self._inner.select_config(contract, state, configs)
        finally:
            self.diagnostics.wall_clock_s.append(time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--mode", choices=("choice", "generate"), default="choice")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    backend = TransformersBackend(args.model_id)
    label = f"llm_{args.mode}"
    runner = MissionRunner(scenario, label, policy=_TimedPolicy(LlmPolicy(backend, mode=args.mode)))
    policy: _TimedPolicy = runner._policy  # the object we just injected
    result = runner.run()

    wall = policy.diagnostics.wall_clock_s
    report = {
        "mission_result": result.to_dict(include_observations=False),
        "policy": {
            "kind": "llm",
            "mode": args.mode,
            **backend.describe(),
            "decisions": policy.diagnostics.decisions,
            "parse_failures": policy.diagnostics.parse_failures,
            "choices": [
                {"frame_id": frame, "config_id": chosen, "raw_reply": reply}
                for frame, chosen, reply in policy.diagnostics.choices
            ],
            "decision_wall_clock_s": {
                "mean": statistics.fmean(wall) if wall else 0.0,
                "max": max(wall) if wall else 0.0,
                "n": len(wall),
                "provenance": (
                    "measured on the development machine around select_config; DIAGNOSTIC "
                    "only — mission time is charged solely by the scenario's "
                    "policy_execution block (this scenario declares "
                    + (
                        f"location={scenario.simulation.policy_execution.location})"
                        if scenario.simulation.policy_execution is not None
                        else "none)"
                    )
                ),
            },
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model_slug = args.model_id.rsplit("/", 1)[-1].lower()
    path = args.output / f"{scenario.scenario_id}_{label}_{model_slug}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    switches = sum(
        1
        for previous, current in zip(
            result.config_selection_history, result.config_selection_history[1:], strict=False
        )
        if previous != current
    )
    print(
        f"{label}: success={result.mission_success} "
        f"recall={result.quality.get('target_recall')} "
        f"battery={result.final_battery_frac:.3f} comm={result.cumulative_communication_mb:.2f}MB "
        f"switches={switches} parse_failures={policy.diagnostics.parse_failures} "
        f"decision_wall_mean={report['policy']['decision_wall_clock_s']['mean']:.2f}s"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
