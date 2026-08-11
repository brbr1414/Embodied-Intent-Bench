# Intent compiler (prototype) — natural language → V1 mission contract

Compiles a natural-language mission instruction into the **frozen V1 `Contract`**
the benchmark already consumes:

```
"20분 안에 끝내. 배터리는 3할 남겨. 원본 영상 전송 금지. 최소 90%는 찾아."
        │  local LLM (greedy) + schema prompt
        ▼
{"deadline_s": 1200, "min_final_battery_frac": 0.3,
 "privacy_level": "features_only", "quality_threshold": 0.9, ...}
        │  frozen load_contract() — the ONLY validator
        ▼
Contract → 기존 policy.select_config(contract, state, configs) 그대로 소비
```

## Position in the architecture

- **Ground-station preprocessing**, like the empirical bundle builder: the
  compiler runs *before* a mission and produces an artifact (contract JSON +
  assumptions + provenance). The benchmark run itself stays deterministic and
  LLM-free; `aerointentbench` never imports this package.
- The LLM decides only the six mission knobs. `task_id`, `evidence_type`, and
  `schema_version` are injected constants; extra keys the model invents are
  dropped and recorded in provenance.
- Every field the intent did not state must appear in the `assumptions` list —
  a compiled contract that hides its guesses is a defect.
- The compiler never sees the scenario, runtime state, ground truth, or results.

## Files

| File | Role |
|---|---|
| `interfaces.py` | Backend protocol, artifact types, frozen-loader validation |
| `compiler.py` | Orchestration: backend → constants → validation → artifact |
| `llm_backend.py` | Local HF model (greedy), schema prompt, JSON extraction |
| `eval_set.json` | 12 hand-written KO/EN paired intents (references authored before any compiler ran; never tune them to a compiler) |
| `run_eval.py` | Compile the set, score stated fields / defaults+assumptions / validation failures |

## Running

```bash
# separate environment; see requirements.txt (torch + transformers)
python -m experiments.intent_compiler.run_eval \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output results/intent_compiler/qwen2.5-1.5b
```

Artifacts land under `results/` (local-only, gitignored): one
`<case_id>.json` per intent (contract + assumptions + provenance incl. raw model
output) and a `summary.json` with the three-way score.

## Scoring semantics

- **stated fields**: intent said it — exact enums; numerics within 1% relative
  (or inside the authored band for deliberately vague phrasing). Misses record
  direction (stricter/looser) because the two directions fail missions
  differently.
- **defaults + assumptions**: intent did not say it — the compiled value must be
  the documented default AND the field must be listed as an assumption. A
  correct-but-silent guess still fails.
- **validation failures**: the frozen loader rejected the output (or no JSON
  parsed). One greedy attempt per intent, no retries, never hidden.

## Honest limitations (v1)

- Determinism is greedy-decoding-on-this-machine, pinned by (model id, prompt
  hash) in provenance — not a cross-platform bit-exactness claim.
- The eval set is 12 hand-written cases: a smoke measure of feasibility, not a
  benchmark result. The reverse-generated dataset (sample contracts → render
  intents → require recovery) and mission-consequence scoring (run compiled vs
  reference contracts through the benchmark) are the planned next stages, as is
  a satisfiability check of compiled contracts via the existing skyline.
- No interactive clarification: missing fields become documented assumptions,
  the compiler never asks back (a candidate later methodology).

## First run (Qwen2.5-1.5B-Instruct, greedy, MPS — 2026-08-11)

`results/intent_compiler/qwen2.5-1.5b/` (local-only). Headline numbers:

| measure | score |
|---|---|
| frozen-loader validation | **12/12** — every output was a legal contract |
| stated fields | **21/24 (87.5%)** |
| defaults + assumption listing | **0/36** |

The three stated-field misses, verbatim:
`ko_full_explicit.deadline_s` 20분 → 1800 (looser — Korean duration mis-scaled);
`ko_local_only.privacy_level` "어떤 데이터도 내보내면 안 돼" → `remote_allowed`
(**the dangerous direction**: a privacy violation compiled into the contract);
`en_remote_allowed.privacy_level` → `features_only` (over-strict, safe direction).

The 0/36 is one failure mode, not thirty-six: the model uses the correct default
VALUES but simply never emits the `assumptions` array (and once drifted the
privacy default to a contextual guess). A 1.5B model handles structured output
and unit grounding; it fails the meta-task of declaring what it guessed.

Implications for the next iteration, in order of leverage: (1) the PIPELINE
methodology — model extracts only fields the text states, deterministic code
fills defaults and derives assumptions — makes assumption listing correct BY
CONSTRUCTION and removes the entire 0/36 axis; (2) privacy-level mapping is the
highest-risk language problem (2 of 3 misses, one in the unsafe direction) and
deserves few-shot examples or a dedicated classification pass; (3) a larger
local model (3B/7B) is the brute-force comparison row.
