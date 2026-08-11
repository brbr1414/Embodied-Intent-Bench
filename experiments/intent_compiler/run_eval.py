"""Run the intent compiler over the paired eval set and score it honestly.

Scoring keeps three error kinds apart, because they mean different things:

- **stated-field errors** — the intent said it, the compiler got it wrong. The
  numbers that matter.
- **default behaviour** — the intent did not say it; the compiled value must be
  the documented default AND the field must appear in the assumptions list.
  A silently-guessed field counts as an assumption failure even if the value
  happens to be right.
- **validation failures** — the frozen loader rejected the compiled contract, or
  no JSON came back at all. Counted per case, never hidden by retries (there are
  none: one greedy attempt per intent).

Numeric scoring: exact reference -> relative tolerance 1%; [lo, hi] band (used
only for deliberately vague phrasing) -> inside the band. Direction is also
recorded for stated-field misses (stricter/looser than intended) because the
two directions fail missions differently.

Usage (from the repository root, inside the experiments environment):
    python -m experiments.intent_compiler.run_eval --model Qwen/Qwen2.5-1.5B-Instruct \
        --output results/intent_compiler/qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.intent_compiler.compiler import compile_intent
from experiments.intent_compiler.interfaces import CompilationResult

logger = logging.getLogger(__name__)

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
_NUMERIC_FIELDS = (
    "quality_threshold",
    "deadline_s",
    "communication_budget_mb",
    "min_final_battery_frac",
)
_KNOBS = (*_NUMERIC_FIELDS, "privacy_level")
_RELATIVE_TOLERANCE = 0.01


def _match_numeric(compiled: float, reference: Any) -> tuple[bool, str]:
    if isinstance(reference, list):
        low, high = float(reference[0]), float(reference[1])
        if low <= compiled <= high:
            return True, "in_band"
        return False, "below_band" if compiled < low else "above_band"
    target = float(reference)
    if target == 0:
        return compiled == 0, "exact"
    if abs(compiled - target) / abs(target) <= _RELATIVE_TOLERANCE:
        return True, "exact"
    return False, "looser" if compiled > target else "stricter"


def score_case(case: dict[str, Any], result: CompilationResult, defaults: dict) -> dict[str, Any]:
    compiled = result.contract
    values = {
        "quality_threshold": compiled.quality_threshold,
        "deadline_s": compiled.deadline_s,
        "communication_budget_mb": compiled.communication_budget_mb,
        "min_final_battery_frac": compiled.min_final_battery_frac,
        "privacy_level": compiled.privacy_level.value,
    }
    assumed_fields = {a.field_name for a in result.assumptions}
    stated = set(case["stated"])
    field_reports: dict[str, dict[str, Any]] = {}
    stated_correct = stated_total = default_correct = default_total = 0

    for field in _KNOBS:
        compiled_value = values[field]
        if field in stated:
            stated_total += 1
            reference = case["reference"][field]
            if field == "privacy_level":
                ok, detail = compiled_value == reference, "enum"
            else:
                ok, detail = _match_numeric(float(compiled_value), reference)
            stated_correct += ok
            field_reports[field] = {
                "kind": "stated",
                "compiled": compiled_value,
                "reference": reference,
                "correct": ok,
                "detail": detail,
                "wrongly_marked_assumed": field in assumed_fields,
            }
        else:
            default_total += 1
            expected = defaults[field]
            value_ok = (
                compiled_value == expected
                if field == "privacy_level"
                else _match_numeric(float(compiled_value), expected)[0]
            )
            assumption_ok = field in assumed_fields
            default_correct += value_ok and assumption_ok
            field_reports[field] = {
                "kind": "default",
                "compiled": compiled_value,
                "expected_default": expected,
                "default_value_ok": value_ok,
                "listed_as_assumption": assumption_ok,
            }
    return {
        "case_id": case["case_id"],
        "validated": True,
        "stated_correct": stated_correct,
        "stated_total": stated_total,
        "default_correct": default_correct,
        "default_total": default_total,
        "fields": field_reports,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="subset of case_ids to run")
    args = parser.parse_args(argv)

    from experiments.intent_compiler.interfaces import prompt_hash
    from experiments.intent_compiler.llm_backend import SYSTEM_PROMPT, LocalLlmBackend

    eval_set = json.loads(_EVAL_SET_PATH.read_text(encoding="utf-8"))
    defaults = eval_set["defaults"]
    cases = [
        case for case in eval_set["cases"] if not args.cases or case["case_id"] in set(args.cases)
    ]
    backend = LocalLlmBackend(model_id=args.model, device=args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures = 0
    for case in cases:
        try:
            result = compile_intent(backend, case["intent"])
        except Exception as error:  # one greedy attempt; a failure is a result
            logger.info("%-22s VALIDATION/PARSE FAILURE: %s", case["case_id"], error)
            reports.append({"case_id": case["case_id"], "validated": False, "error": str(error)})
            failures += 1
            continue
        report = score_case(case, result, defaults)
        reports.append(report)
        artifact_path = args.output / f"{case['case_id']}.json"
        artifact_path.write_text(
            json.dumps(result.to_artifact(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "%-22s stated %d/%d  defaults(+assumption) %d/%d",
            case["case_id"],
            report["stated_correct"],
            report["stated_total"],
            report["default_correct"],
            report["default_total"],
        )

    scored = [r for r in reports if r.get("validated")]
    summary = {
        "model": args.model,
        "prompt_hash": prompt_hash(SYSTEM_PROMPT),
        "cases": len(cases),
        "validation_failures": failures,
        "stated_field_accuracy": {
            "correct": sum(r["stated_correct"] for r in scored),
            "total": sum(r["stated_total"] for r in scored),
        },
        "default_and_assumption_accuracy": {
            "correct": sum(r["default_correct"] for r in scored),
            "total": sum(r["default_total"] for r in scored),
        },
        "reports": reports,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stated = summary["stated_field_accuracy"]
    defaults_acc = summary["default_and_assumption_accuracy"]
    logger.info(
        "\nTOTAL  validated %d/%d  stated fields %d/%d  defaults+assumptions %d/%d  -> %s",
        len(scored),
        len(cases),
        stated["correct"],
        stated["total"],
        defaults_acc["correct"],
        defaults_acc["total"],
        args.output / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
