"""Local-LLM backend for the intent compiler (HF transformers, greedy decoding).

Model calls are LOCAL (no API): a small instruct model runs on this machine via
``transformers``. Determinism policy: greedy decoding (``do_sample=False``) with
a pinned model id — for a fixed (model id, prompt) pair the output is stable on
a given machine/library version, and the full identity (model, prompt hash,
generation config) is recorded in provenance so any artifact can be audited.

The prompt teaches the FROZEN schema and the grounding rules; the model returns
one JSON object. Everything the model must not decide (task_id, evidence_type,
schema_version) is injected afterwards from ``SCHEMA_CONSTANTS`` — the model is
never allowed to touch them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from experiments.intent_compiler.interfaces import SCHEMA_CONSTANTS

__all__ = ["SYSTEM_PROMPT", "LocalLlmBackend", "extract_json_object"]

SYSTEM_PROMPT: Final = """\
You compile a natural-language UAV mission instruction into a strict JSON mission
contract for a search mission. Output ONE JSON object and nothing else.

The JSON object must have exactly these keys:

{
  "contract_id": string, short SCREAMING_SNAKE_CASE id derived from the intent,
  "quality_metric": "target_recall",
  "quality_operator": ">=",
  "quality_threshold": number in [0,1] — required fraction of targets found,
  "deadline_s": number > 0 — mission time limit in SECONDS,
  "communication_budget_mb": number >= 0 — total uplink+downlink allowance in MB,
  "min_final_battery_frac": number in [0,1] — battery that must REMAIN at the end,
  "privacy_level": one of "local_only" | "features_only" | "remote_allowed",
  "assumptions": list of {"field": string, "reason": string} — one entry for EVERY
                 field above whose value the instruction did not state
}

Grounding rules:
- Convert all durations to seconds ("30 minutes" -> 1800, "한 시간 안에" -> 3600).
- Battery phrases give the REMAINING fraction ("keep 20% for the return trip",
  "배터리 2할은 남겨" -> 0.2).
- privacy_level: "local_only" when NOTHING may be transmitted off the vehicle;
  "features_only" when imagery/raw video must not leave but derived data may
  ("영상은 내보내지 마", "no raw footage off the drone"); "remote_allowed" when
  offloading to a server is explicitly fine or clearly implied.
- quality_threshold: explicit fractions/percentages win; otherwise ground
  "find them all / 반드시 다 찾아" near 0.9, "thorough / 웬만하면 다" near 0.75,
  "quick look / 대충 훑어" near 0.5.
- Unstated fields get these defaults AND an assumptions entry:
  quality_threshold 0.7, deadline_s 900, communication_budget_mb 400,
  min_final_battery_frac 0.2, privacy_level "features_only".
- Never invent stricter or looser values than the instruction supports.
"""

_GENERATION: Final = {"max_new_tokens": 512, "do_sample": False}


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from model output.

    Small local models sometimes wrap the object in code fences or prose; the
    object itself must still parse strictly — no repair beyond locating it.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start < 0:
            raise ValueError(f"no JSON object in model output: {text[:200]!r}")
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    break
        if candidate is None:
            raise ValueError(f"unbalanced JSON object in model output: {text[:200]!r}")
    return json.loads(candidate)


class LocalLlmBackend:
    """Greedy local-generation backend over a pinned HF instruct model."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the LLM backend needs the experiments/intent_compiler/requirements.txt "
                "environment (torch + transformers); it is never part of the benchmark core"
            ) from error

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)
        self._model.eval()
        self.model_id = model_id
        self.backend_id = f"local_llm:{model_id}"

    def compile(self, intent_text: str) -> tuple[dict[str, Any], list[dict[str, str]], str]:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": intent_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **_GENERATION,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        parsed = extract_json_object(raw)
        assumptions = parsed.pop("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        fields = {**SCHEMA_CONSTANTS, **parsed}
        return fields, assumptions, raw

    @property
    def generation_config(self) -> dict[str, Any]:
        return dict(_GENERATION)
