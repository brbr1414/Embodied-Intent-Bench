"""The compiler pipeline: backend output -> frozen-loader validation -> artifact.

One deliberately thin orchestration layer, so the division of authority stays
visible: the BACKEND interprets language, ``SCHEMA_CONSTANTS`` supplies what the
model may not decide, and the FROZEN V1 LOADER is the only validator. A compile
succeeds if and only if the benchmark itself would accept the contract.
"""

from __future__ import annotations

from typing import Any

from experiments.intent_compiler.interfaces import (
    CompilationResult,
    CompiledAssumption,
    IntentCompilerBackend,
    validate_compiled_contract,
)

__all__ = ["compile_intent"]

#: The knobs the language model is supposed to decide. Anything else it emits
#: (beyond contract_id and the injected constants) is discarded and recorded —
#: silently keeping extra keys would let the model widen the schema.
_MODEL_DECIDED_FIELDS = (
    "quality_metric",
    "quality_operator",
    "quality_threshold",
    "deadline_s",
    "communication_budget_mb",
    "min_final_battery_frac",
    "privacy_level",
)


def compile_intent(backend: IntentCompilerBackend, intent_text: str) -> CompilationResult:
    """Compile one intent through the backend and the frozen validator."""
    fields, raw_assumptions, raw_output = backend.compile(intent_text)

    payload: dict[str, Any] = {}
    dropped: list[str] = []
    allowed = set(_MODEL_DECIDED_FIELDS) | {
        "schema_version",
        "task_id",
        "evidence_type",
        "contract_id",
    }
    for key, value in fields.items():
        if key in allowed:
            payload[key] = value
        else:
            dropped.append(key)

    contract = validate_compiled_contract(payload)

    assumptions = tuple(
        CompiledAssumption(
            field_name=str(entry.get("field", "?")),
            assumed_value=payload.get(str(entry.get("field", "?"))),
            reason=str(entry.get("reason", "")),
        )
        for entry in raw_assumptions
        if isinstance(entry, dict)
    )
    provenance = {
        "compiler": "experiments/intent_compiler",
        "backend_id": backend.backend_id,
        "intent_text": intent_text,
        "dropped_model_keys": dropped,
        "raw_model_output": raw_output,
    }
    return CompilationResult(
        contract=contract,
        contract_payload=payload,
        assumptions=assumptions,
        provenance=provenance,
    )
