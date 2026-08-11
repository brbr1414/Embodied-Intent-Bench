"""Intent compiler interfaces: natural language in, a validated V1 Contract out.

The compiler is a GROUND-STATION PREPROCESSING step, like the empirical bundle
builder: it runs before the mission, produces an artifact (a contract JSON that
the frozen V1 loader accepts, plus provenance), and the benchmark itself remains
deterministic and LLM-free. Nothing in ``aerointentbench`` imports this package.

Boundaries:

- The compiler sees the INTENT TEXT and the platform-independent schema rules —
  never the scenario, runtime state, ground truth, or benchmark results. It
  compiles what the operator asked for, not what would score well.
- The LLM decides only the six mission knobs (quality metric/operator/threshold,
  deadline, communication budget, battery floor, privacy level). ``task_id`` and
  ``evidence_type`` are V1 constants — there is one task family.
- Every field the intent did not state must be listed as an ASSUMPTION with the
  default used; a compiled contract that hides its guesses is a defect.
- Output validation is the frozen loader (``load_contract``), strict by
  construction: unknown fields, bad enums, and out-of-range values fail loudly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from aerointentbench.schemas.contract import Contract, load_contract

__all__ = [
    "SCHEMA_CONSTANTS",
    "CompilationResult",
    "CompiledAssumption",
    "IntentCompilerBackend",
    "validate_compiled_contract",
]

#: V1 has exactly one task family; these are constants of the compile target, not
#: decisions the language model is allowed to make.
SCHEMA_CONSTANTS: dict[str, str] = {
    "schema_version": "1.0",
    "task_id": "HUMAN_SEARCH_SEGMENTATION",
    "evidence_type": "instance_mask_set",
}


class IntentCompilerBackend(Protocol):
    """One way of turning intent text into the raw contract-field dictionary.

    Returns ``(fields, assumptions, raw_output)``: the six mission knobs plus a
    ``contract_id``, the list of assumption records for unstated fields, and the
    backend's raw output for provenance/debugging. Implementations must be
    deterministic for a fixed (backend identity, intent) pair.
    """

    backend_id: str

    def compile(self, intent_text: str) -> tuple[dict[str, Any], list[dict[str, str]], str]: ...


@dataclass(frozen=True, slots=True)
class CompiledAssumption:
    """One field the compiler had to guess because the intent did not state it."""

    field_name: str
    assumed_value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "assumed_value": self.assumed_value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """A compiled contract artifact: the contract, its guesses, and its provenance."""

    contract: Contract
    contract_payload: dict[str, Any]
    assumptions: tuple[CompiledAssumption, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_artifact(self) -> dict[str, Any]:
        """The JSON artifact a mission run consumes (contract) and an audit reads (rest)."""
        return {
            "contract": self.contract_payload,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "provenance": self.provenance,
        }


def validate_compiled_contract(payload: dict[str, Any]) -> Contract:
    """Round-trip the compiled payload through the FROZEN V1 loader.

    The loader is the single validation authority — this function adds nothing to
    it, so the compiler cannot drift from what the benchmark actually accepts.
    """
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "compiled_contract.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_contract(path)


def prompt_hash(prompt: str) -> str:
    """A stable identity for the exact prompt text used, for provenance."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
