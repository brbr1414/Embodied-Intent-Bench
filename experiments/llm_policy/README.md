# LLM-as-Policy

A local instruction-tuned LLM as the configuration-selection policy, on the frozen V1
`Policy` protocol. The model sees exactly what every other policy sees — contract,
frozen `RuntimeState`, allowed configurations with public profiles — serialized into a
deterministic prompt. No ground truth, no network trace, no simulator internals.

## Constraining the action space

The action is one `config_id` from a finite set, and the two modes differ in how that
constraint is enforced:

- **`choice`** (default): the backend scores each allowed id as a continuation of the
  prompt (length-normalized token log-likelihood, greedy, no sampling) and returns the
  argmax. Invalid actions are impossible by construction; selection is deterministic
  for a fixed model and device.
- **`generate`**: greedy free-form generation, then strict parsing for an allowed id.
  A reply that names no allowed option is a **parse failure**: the policy falls back to
  the first catalog row and the failure is counted and reported — a model that cannot
  keep the output format is a finding, not an exception to hide.

## Boundaries

- Heavy deps (torch/transformers) live in `backend.py` only, imported lazily. The CI
  tests (`tests/test_llm_policy.py`) inject a fake backend and never load a model.
- The policy enters the mission through `MissionRunner(..., policy=...)` — a
  composition-root injection seam. Nothing is registered in the core policy registry.
- Decision wall-clock is measured by the runner script as a **diagnostic**; mission
  time is charged only by the scenario's `policy_execution` block (docs/v2_design.md
  §10.12). Grounding that block for an LLM policy needs a board measurement of the
  model — the Mac numbers here are not that measurement.

## Running

```bash
~/.venvs/sc2bench/bin/python -m experiments.llm_policy.run_llm_mission \
  --scenario data/v2_scenarios/demo_img1_model_catalog_xavier.json \
  --mode choice --output results/llm_policy/
```

Reports land in `results/llm_policy/` (local-only, gitignored): mission result plus
model id/device, per-decision choices with raw replies, parse-failure count, and
decision wall-clock stats.

## Status

Prototype. Default model `Qwen/Qwen2.5-1.5B-Instruct` (cached locally by the
intent-compiler work). Results on the catalog scenarios are recorded in
`docs/v3x_extensions.md`; treat them as behaviour of one small model on one prompt
design, never as "LLMs can/cannot do this".
