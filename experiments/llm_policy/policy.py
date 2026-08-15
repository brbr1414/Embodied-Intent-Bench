"""The LLM policy: prompt construction, constrained selection, and honest bookkeeping.

Two ways to constrain the model to the finite action space:

- ``choice`` (default): the backend scores each allowed ``config_id`` as a continuation
  of the prompt and returns the argmax. An invalid action is impossible by
  construction, and greedy scoring is deterministic for a fixed model.
- ``generate``: the backend generates freely (greedy, short) and this policy parses the
  first allowed config_id out of the text. A reply that names no allowed option is a
  PARSE FAILURE: the policy falls back to the safest declared option and counts the
  failure — invalid outputs are a finding about the model, not an exception to hide.

The prompt is a pure function of (contract, state, configs, public profiles): same
inputs, same bytes. It contains policy-visible information ONLY — adding anything
ground-truth-derived here would invalidate the benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from aerointentbench.policies.base import estimated_latency_s
from aerointentbench.schemas.configuration import ConfigCatalog, Configuration
from aerointentbench.schemas.contract import Contract
from aerointentbench.schemas.profile import PublicProfileView
from aerointentbench.schemas.runtime_state import RuntimeState

__all__ = ["LlmBackend", "LlmPolicy", "PolicyDiagnostics"]


class LlmBackend(Protocol):
    """What the policy needs from a language model. Implementations may be fakes."""

    def choose(self, prompt: str, options: list[str]) -> str:
        """Return the option with the highest continuation likelihood."""
        ...

    def generate(self, prompt: str) -> str:
        """Return a short free-form completion (greedy)."""
        ...


@dataclass
class PolicyDiagnostics:
    """Per-mission bookkeeping the experiment report reads. Never mission state."""

    decisions: int = 0
    parse_failures: int = 0
    #: Wall-clock seconds per decision, measured by the experiment runner (diagnostic
    #: only — mission time comes from the scenario's policy_execution block, if any).
    wall_clock_s: list[float] = field(default_factory=list)
    #: (frame_id, chosen config_id, raw reply or None) per decision.
    choices: list[tuple[int, str, str | None]] = field(default_factory=list)


class LlmPolicy:
    """V1 ``Policy`` protocol over an injected LLM backend. Stateful per mission."""

    def __init__(
        self,
        backend: LlmBackend,
        *,
        mode: str = "choice",
        public_profiles: PublicProfileView | None = None,
    ) -> None:
        if mode not in ("choice", "generate"):
            raise ValueError(f"mode {mode!r} is not 'choice' or 'generate'")
        self._backend = backend
        self._mode = mode
        self._profiles = public_profiles
        self.diagnostics = PolicyDiagnostics()

    # -- V1 Policy protocol -------------------------------------------------------------

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        options = [config.config_id for config in configs]
        prompt = self.build_prompt(contract, state, configs)
        self.diagnostics.decisions += 1
        if self._mode == "choice":
            chosen = self._backend.choose(prompt, options)
            if chosen not in options:  # a broken backend, not a broken model
                raise ValueError(f"backend chose {chosen!r}, not one of the offered options")
            self.diagnostics.choices.append((state.frame_id, chosen, None))
            return chosen
        reply = self._backend.generate(prompt)
        chosen = self._parse(reply, options)
        if chosen is None:
            self.diagnostics.parse_failures += 1
            chosen = options[0]  # catalog order: the scenario's own first (safest) row
        self.diagnostics.choices.append((state.frame_id, chosen, reply))
        return chosen

    # -- prompt -------------------------------------------------------------------------

    def build_prompt(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        """Deterministic serialization of the policy-visible surface, nothing else."""
        network = state.network
        lines = [
            "You select the perception inference configuration for a search drone, "
            "once per second.",
            "Mission contract:",
            f"- quality: {contract.quality_metric} {contract.quality_operator.value} "
            f"{contract.quality_threshold:.2f}",
            f"- deadline: {contract.deadline_s:.0f} s total",
            f"- battery: final charge must stay above {contract.min_final_battery_frac:.2f}",
            f"- communication budget: {contract.communication_budget_mb:.1f} MB uplink total",
            f"- privacy level: {contract.privacy_level.value}",
            "Current state:",
            f"- time {state.current_time_s:.1f} s, deadline remaining "
            f"{state.remaining_deadline_s:.1f} s, path progress {state.path_progress:.2f}",
            f"- battery {state.battery_frac:.3f}",
            f"- communication used {state.cumulative_communication_mb:.2f} MB of "
            f"{contract.communication_budget_mb:.1f} MB",
            f"- network: bandwidth {network.bandwidth_mbps:.1f} Mbps, rtt "
            f"{network.rtt_ms:.0f} ms, packet loss {network.packet_loss_frac:.2f}"
            + (" (LINK DOWN)" if network.is_disconnected else ""),
            f"- current configuration: {state.current_config_id or 'none yet'}",
            f"- evidence so far: {state.evidence_summary.predicted_unique_targets} predicted "
            f"targets in {state.evidence_summary.processed_frames} processed frames",
            "Options (choose exactly one id):",
        ]
        for config in configs:
            lines.append(self._describe(config, state))
        lines.append(
            "A remote option on a dead link will fail; an option forbidden by the privacy "
            "level is a recorded violation. Reply with one option id only."
        )
        lines.append("Answer:")
        return "\n".join(lines)

    def _describe(self, config: Configuration, state: RuntimeState) -> str:
        profile = None if self._profiles is None else self._profiles.get(config.config_id)
        placement = "remote" if config.strategy.is_remote else "onboard"
        if profile is None:
            return f"- {config.config_id}: {placement}, profile not disclosed"
        latency = estimated_latency_s(config, profile, state.network)
        latency_text = "unreachable" if latency is None else f"~{latency * 1000.0:.0f} ms"
        return (
            f"- {config.config_id}: {placement}, quality tier {profile.quality_tier}, "
            f"estimated latency {latency_text}, upload {profile.expected_upload_mb:.3f} MB/frame"
        )

    # -- parsing ------------------------------------------------------------------------

    @staticmethod
    def _parse(reply: str, options: list[str]) -> str | None:
        """First allowed id appearing as a whole token in the reply, else None."""
        for match in re.finditer(r"[A-Za-z0-9_.\-]+", reply):
            if match.group(0) in options:
                return match.group(0)
        return None
