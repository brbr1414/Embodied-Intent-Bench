"""LLM-as-Policy adapter: constrained action space, honest parse failures, injection.

CI-safe: a fake backend stands in for the model (no torch/transformers import), the V2
mission tests reuse the tiny PNG world. What is pinned: the prompt is a deterministic
pure function of the policy-visible surface and offers exactly the allowed options;
choice mode can only ever act inside the option set; generate mode counts unparseable
replies and falls back instead of crashing; the MissionRunner injection seam runs a
constructed policy object and labels the result with the given name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.v2.runner import MissionRunner
from aerointentbench.v2.scenario import load_scenario
from experiments.llm_policy.policy import LlmPolicy
from tests.test_v2_visual_loop import scenario_payload, write_world_png


class ScriptedBackend:
    """Returns queued replies (generate) or a fixed preference order (choose)."""

    def __init__(self, *, prefer: str | None = None, replies: list[str] | None = None) -> None:
        self._prefer = prefer
        self._replies = list(replies or [])
        self.prompts: list[str] = []

    def choose(self, prompt: str, options: list[str]) -> str:
        self.prompts.append(prompt)
        if self._prefer in options:
            return self._prefer
        return options[0]

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else ""


def _mission(tmp_path: Path, policy: LlmPolicy, label: str = "llm_test"):
    world = write_world_png(tmp_path / "world.png")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario_payload(world)))
    return MissionRunner(load_scenario(path), label, policy=policy).run()


class TestPromptSurface:
    def test_prompt_is_a_pure_function_and_offers_every_option(self, tmp_path: Path) -> None:
        backend = ScriptedBackend(prefer="STRONG")
        policy = LlmPolicy(backend)
        _mission(tmp_path, policy)
        assert backend.prompts, "the policy should have been consulted"
        first = backend.prompts[0]
        assert "FAST" in first and "STRONG" in first
        assert "privacy level" in first and "battery" in first
        # Re-running the identical mission reproduces the identical first prompt.
        backend2 = ScriptedBackend(prefer="STRONG")
        _mission(tmp_path, LlmPolicy(backend2))
        assert backend2.prompts[0] == first


class TestChoiceMode:
    def test_backend_preference_drives_the_mission(self, tmp_path: Path) -> None:
        policy = LlmPolicy(ScriptedBackend(prefer="STRONG"))
        result = _mission(tmp_path, policy)
        assert set(result.config_selection_history) == {"STRONG"}
        assert policy.diagnostics.decisions == len(result.config_selection_history)
        assert policy.diagnostics.parse_failures == 0

    def test_a_backend_answer_outside_the_options_is_recorded_not_absorbed(
        self, tmp_path: Path
    ) -> None:
        class BrokenBackend(ScriptedBackend):
            def choose(self, prompt: str, options: list[str]) -> str:
                return "NOT_AN_OPTION"

        # A backend ignoring its option set is broken plumbing: the policy raises, the
        # runner records a policy failure per slot (V1 semantics — invalid actions are
        # data, not crashes), and every slot runs the scenario fallback, visibly.
        policy = LlmPolicy(BrokenBackend())
        result = _mission(tmp_path, policy)
        assert set(result.config_selection_history) == {"FAST"}
        assert all(not log.action_valid for log in result.observations)


class TestGenerateMode:
    def test_valid_reply_is_parsed_out_of_prose(self, tmp_path: Path) -> None:
        replies = ["I will pick STRONG because the targets are small."] * 64
        policy = LlmPolicy(ScriptedBackend(replies=replies), mode="generate")
        result = _mission(tmp_path, policy)
        assert set(result.config_selection_history) == {"STRONG"}
        assert policy.diagnostics.parse_failures == 0

    def test_unparseable_reply_falls_back_and_is_counted(self, tmp_path: Path) -> None:
        replies = ["Hmm, tough call, maybe the light one?"] * 64
        policy = LlmPolicy(ScriptedBackend(replies=replies), mode="generate")
        result = _mission(tmp_path, policy)
        # Catalog order: FAST is the first (safest) declared configuration.
        assert set(result.config_selection_history) == {"FAST"}
        assert policy.diagnostics.parse_failures == policy.diagnostics.decisions > 0
        # Raw replies are preserved for the report — failures are findings, not noise.
        assert policy.diagnostics.choices[0][2] is not None


class TestInjectionSeam:
    def test_injected_policy_is_used_and_the_label_names_the_result(self, tmp_path: Path) -> None:
        policy = LlmPolicy(ScriptedBackend(prefer="STRONG"))
        result = _mission(tmp_path, policy, label="llm_choice")
        assert result.policy_name == "llm_choice"
        assert set(result.config_selection_history) == {"STRONG"}

    def test_without_injection_the_registry_path_is_unchanged(self, tmp_path: Path) -> None:
        world = write_world_png(tmp_path / "world.png")
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(scenario_payload(world)))
        result = MissionRunner(load_scenario(path), "always_fast").run()
        assert set(result.config_selection_history) == {"FAST"}
