"""Numeric mission metrics: contract margins and the skyline regret.

Responsibility and boundaries: this module RESTATES what the evaluator and runner
already computed — it never re-decides success. Three tiers, by the question they
answer:

- ``mission_success`` (unchanged, computed by the runner): did the mission keep the
  contract?
- **Contract margins** (here): by how much? One signed, normalized slack per
  constraint axis, denominated by the contract itself (no tunable weights):
  battery ``(final - floor) / (1 - floor)``, communication
  ``(budget - used) / budget``, deadline ``(deadline - t) / deadline``, quality
  ``(value - threshold) / (1 - threshold)`` for ``>=`` contracts (mirrored for
  ``<=``), privacy ``0`` when clean else ``-violations / processed``. The headline
  scalar is ``min_margin`` — the tightest axis — and ``min_margin >= 0`` coincides
  with ``mission_success`` by construction; :func:`verify_margins` enforces that
  axis-by-axis against the evaluator's booleans and fails loudly on divergence.
- **Skyline regret** (here): how far from the best achievable? ``skyline recall -
  achieved recall``, defined for ``target_recall`` contracts only (the skyline's own
  scope). The skyline is GT-aware and does not model policy_execution decision
  costs; quote regret with that asymmetry in mind.

Evaluation stance (documented, deliberate): the contract defines sufficiency.
Among successful missions, larger margins are better; surplus quality beyond the
threshold appears in the quality margin, never as a separate score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aerointentbench.schemas.common import ComparisonOperator
from aerointentbench.schemas.contract import Contract

if TYPE_CHECKING:
    from aerointentbench.v2.runner import V2MissionResult

__all__ = [
    "MarginReport",
    "contract_margins",
    "delivery_delays",
    "skyline_regret",
    "timely_recall",
    "verify_margins",
]

#: Margin axis names, in the constraint order the runner reports.
AXES = ("quality", "deadline", "battery", "communication", "privacy")


#: Axes that participate in ``min_margin`` only when NEGATIVE. Privacy is binary — a
#: clean mission has no meaningful slack to compare. Deadline is a design-owned sanity
#: bound on the fixed-path missions this benchmark runs: with a predefined trajectory
#: at fixed speed, the deadline margin is a path-determined constant identical across
#: policies (verified: every grand-tour policy ties at +0.083), so letting it bind the
#: minimum would mask the axes a policy actually controls. Mission time itself stays
#: reported numerically (margin vector + final_time_s); a violated deadline still
#: forces the minimum negative, exactly like a privacy violation.
_SANITY_AXES = ("privacy", "deadline")


@dataclass(frozen=True, slots=True)
class MarginReport:
    """Signed normalized slack per constraint axis, plus the binding minimum.

    ``min_margin`` runs over the policy-sensitive axes; sanity axes (privacy,
    deadline — see ``_SANITY_AXES``) participate only when violated, guaranteeing the
    minimum goes negative exactly when the mission fails.
    """

    margins: dict[str, float]

    def _participating(self) -> dict[str, float]:
        return {
            axis: value
            for axis, value in self.margins.items()
            if axis not in _SANITY_AXES or value < 0.0
        }

    @property
    def min_margin(self) -> float:
        participating = self._participating()
        return min(participating.values())

    @property
    def binding_axis(self) -> str:
        participating = self._participating()
        return min(participating, key=participating.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "margins": {axis: self.margins[axis] for axis in AXES},
            "min_margin": self.min_margin,
            "binding_axis": self.binding_axis,
        }


def _ratio(numerator: float, denominator: float) -> float:
    """Normalized slack; falls back to the raw difference when the contract leaves
    no normalizing range (degenerate but legal specifications)."""
    return numerator / denominator if denominator > 0.0 else numerator


def _quality_margin(contract: Contract, value: float) -> float:
    operator = contract.quality_operator
    threshold = contract.quality_threshold
    if operator is ComparisonOperator.GREATER_EQUAL:
        return _ratio(value - threshold, 1.0 - threshold)
    if operator is ComparisonOperator.LESS_EQUAL:
        return _ratio(threshold - value, threshold)
    raise ValueError(
        f"margin semantics are defined for non-strict operators only; contract "
        f"{contract.contract_id!r} uses {operator.value!r} (a zero margin would be "
        "ambiguous between success and failure)"
    )


def contract_margins(
    contract: Contract,
    *,
    quality_value: float,
    final_time_s: float,
    final_battery_frac: float,
    communication_mb: float,
    privacy_violation_count: int,
    processed_observation_count: int,
) -> MarginReport:
    """Margins from the exact quantities the runner's constraint checks consumed."""
    return MarginReport(
        margins={
            "quality": _quality_margin(contract, quality_value),
            "deadline": _ratio(contract.deadline_s - final_time_s, contract.deadline_s),
            "battery": _ratio(
                final_battery_frac - contract.min_final_battery_frac,
                1.0 - contract.min_final_battery_frac,
            ),
            "communication": _ratio(
                contract.communication_budget_mb - communication_mb,
                contract.communication_budget_mb,
            ),
            "privacy": (
                0.0
                if privacy_violation_count == 0
                else -privacy_violation_count / max(1, processed_observation_count)
            ),
        }
    )


_AXIS_TO_CONSTRAINT = {
    "quality": "quality_success",
    "deadline": "deadline_success",
    "battery": "battery_constraint_success",
    "communication": "communication_constraint_success",
    "privacy": "privacy_constraint_success",
}


def verify_margins(report: MarginReport, constraints: dict[str, bool]) -> None:
    """Fail loudly if any margin's sign disagrees with the evaluator's verdict.

    Margins are a restatement; a divergence means this module is wrong, and a wrong
    restatement must never ship silently (same rule as the replay exporter).
    """
    for axis, key in _AXIS_TO_CONSTRAINT.items():
        margin_ok = report.margins[axis] >= 0.0
        if margin_ok != constraints[key]:
            raise AssertionError(
                f"margin restatement diverged from the evaluator on {axis!r}: "
                f"margin {report.margins[axis]:+.6f} vs {key}={constraints[key]}"
            )


def delivery_delays(result: V2MissionResult) -> dict[str, float | None]:
    """Per-target operator-delivery delay: first_delivered - first_visible (seconds).

    ``None`` = the target never reached the operator (never matched, or every
    evidence transmission carrying it was lost). Loud error when the scenario had no
    evidence layer — without one, "delivered" is undefined and a silent 0-delay
    answer would be a lie.
    """
    ledger = result.target_timeliness
    if not ledger.get("available"):
        raise ValueError(
            "operator timeliness is undefined for this mission: "
            + ledger.get("note", "no target_timeliness ledger present")
        )
    delays: dict[str, float | None] = {}
    for target_id, times in ledger["targets"].items():
        visible = times.get("first_visible_s")
        arrived = times.get("first_delivered_s")
        delays[target_id] = None if visible is None or arrived is None else arrived - visible
    return delays


def timely_recall(result: V2MissionResult, horizon_s: float, total_targets: int) -> float:
    """Fraction of ALL mission targets delivered to the operator within ``horizon_s``.

    The denominator is the scenario's total target count (recall discipline), so
    targets that never entered the footprint or were never delivered count against
    the score. TR(T) is monotone in T; report it as a curve over several horizons
    rather than committing to one threshold — the constraint form, if any, comes
    later and from operational grounds.
    """
    delays = delivery_delays(result)
    timely = sum(1 for delay in delays.values() if delay is not None and delay <= horizon_s)
    return timely / max(1, total_targets)


def skyline_regret(skyline_best_recall: float, result: V2MissionResult) -> float:
    """``skyline recall - achieved recall`` for a target_recall mission.

    The skyline is a GT-aware offline upper bound (never a policy result), so regret
    reads as "recall left on the table relative to what was achievable in
    principle". Soundness of the skyline makes this non-negative up to evaluator
    determinism. Loud error off target_recall contracts — no silent metric reuse.
    """
    metric = result.quality.get("quality_metric")
    if metric != "target_recall":
        raise ValueError(
            f"skyline regret is defined for target_recall contracts only, got {metric!r}"
        )
    return skyline_best_recall - float(result.quality["target_recall"])
