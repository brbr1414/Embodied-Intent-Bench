#!/usr/bin/env python3
"""LLM policy decision cost on the Jetson: latency + rail energy per decision.

Measures what experiments/llm_policy does per select_config call, on the board:
- choice mode: length-normalized log-likelihood scoring of every option id
  (one forward per option — the mode's real cost structure);
- generate mode: greedy short generation.

The prompt is a representative mission prompt (same template family and token
scale as the grand-tour missions; 20 option ids from the extended catalog).
Decision cost depends on token counts, not semantics — recorded in provenance.
Protocol mirrors measure.py: idle baseline, warmup, timed reps with INA3221
sampling; marginal = run minus same-invocation idle. Python 3.8 compatible.

Usage: measure_llm_policy.py <model_id> <mode> <mode_label> [out_dir]
"""

import json
import statistics
import sys
import time
from pathlib import Path

from measure import IDLE_S, Sampler, device_string, dist

OPTIONS = [
    "local_light_fp32",
    "local_light_fp16",
    "local_strong_fp32",
    "local_strong_fp16",
    "remote_strong_raw",
    "presplit_es_b064",
    "presplit_es_b512",
    "presplit_ghnd_bq3",
    "maskrcnn_onboard_full",
    "presplit_maskrcnn_fcm",
    "local_strong_fp32_512",
    "local_strong_fp16_512",
    "local_light_fp32_768",
    "local_light_fp16_768",
    "local_dlv3plus_w8a8",
    "local_dlv3plus_fp32_onnx",
    "local_segformer_w8a8",
    "local_segformer_fp32_onnx",
    "local_ffnet40s_w8a8",
    "local_ffnet40s_fp32_onnx",
]

PROMPT = (
    "You select the perception inference configuration for a search drone, once per second.\n"
    "Mission contract:\n"
    "- quality: target_recall >= 0.60\n- deadline: 479 s total\n"
    "- battery: final charge must stay above 0.17\n"
    "- communication budget: 80.0 MB uplink total\n- privacy level: features_only\n"
    "Current state:\n"
    "- time 187.0 s, deadline remaining 292.3 s, path progress 0.43\n"
    "- battery 0.612\n- communication used 0.31 MB of 80.0 MB\n"
    "- network: bandwidth 0.0 Mbps, rtt 0 ms, packet loss 1.00 (LINK DOWN)\n"
    "- current configuration: local_strong_fp16\n"
    "- evidence so far: 6 predicted targets in 187 processed frames\n"
    "Options (choose exactly one id):\n"
    + "\n".join(
        f"- {o}: onboard, quality tier medium, estimated latency ~170 ms, upload 0.000 MB/frame"
        for o in OPTIONS
    )
    + "\nA remote option on a dead link will fail; an option forbidden by the privacy "
    "level is a recorded violation. Reply with one option id only.\nAnswer:"
)

N_DECISIONS = 8


def main():
    model_id, mode, mode_label = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = Path(sys.argv[4] if len(sys.argv) > 4 else "results_llm")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to("cuda").eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
    option_ids = [tokenizer(" " + o, add_special_tokens=False).input_ids for o in OPTIONS]

    def decide_choice():
        best, best_score = None, None
        with torch.no_grad():
            for opt, ids in zip(OPTIONS, option_ids, strict=False):
                full = torch.tensor([prompt_ids + ids], device="cuda")
                logits = model(full).logits[0]
                total = 0.0
                for position, token_id in enumerate(ids):
                    step = len(prompt_ids) + position - 1
                    total += float(torch.log_softmax(logits[step], dim=-1)[token_id])
                score = total / len(ids)
                if best_score is None or score > best_score:
                    best, best_score = opt, score
        return best

    def decide_generate():
        # Manual greedy loop: the board torch build lacks torch.distributed, which
        # transformers' generate() imports. Same compute as greedy generation.
        with torch.no_grad():
            ids = torch.tensor([prompt_ids], device="cuda")
            out = model(ids, use_cache=True)
            past = out.past_key_values
            token = out.logits[0, -1].argmax().view(1, 1)
            produced = [int(token)]
            for _ in range(23):
                out = model(token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = out.logits[0, -1].argmax().view(1, 1)
                produced.append(int(token))
                if produced[-1] == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(produced, skip_special_tokens=True)

    decide = decide_choice if mode == "choice" else decide_generate

    idle = Sampler()
    idle.start()
    time.sleep(IDLE_S)
    idle.halt()
    idle_power = [s["total_w"] for s in idle.samples]

    for _ in range(2):
        decide()
    torch.cuda.synchronize()

    run = Sampler()
    run.start()
    latencies = []
    wall0 = time.perf_counter()
    for _ in range(N_DECISIONS):
        t0 = time.perf_counter()
        decide()
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    wall = time.perf_counter() - wall0
    run.halt()
    run_power = [s["total_w"] for s in run.samples]

    idle_mean = statistics.fmean(idle_power)
    run_mean = statistics.fmean(run_power)
    total_j = run_mean * wall / N_DECISIONS
    marginal_j = (run_mean - idle_mean) * wall / N_DECISIONS
    record = {
        "device": device_string(),
        "mode_label": mode_label,
        "model_id": model_id,
        "decision_mode": mode,
        "prompt_tokens": len(prompt_ids),
        "options": len(OPTIONS),
        "decisions": N_DECISIONS,
        "latency_s": dist(latencies),
        "power_idle_w": dist(idle_power),
        "power_run_w": dist(run_power),
        "energy_total_j_per_decision": total_j,
        "energy_marginal_j_per_decision": marginal_j,
        "provenance": (
            "measured: INA3221 rails ~20 Hz; representative mission prompt (template and "
            "token scale of the grand-tour missions, 20 option ids); decision cost depends "
            "on token counts, not semantics; fp16 CUDA via transformers 4.46"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = model_id.rsplit("/", 1)[-1].lower()
    path = out_dir / f"{mode_label}__{name}__{mode}.json"
    path.write_text(json.dumps(record, indent=1))
    print(
        f"{mode_label} {name} {mode}: {statistics.fmean(latencies):.2f}s/decision "
        f"run={run_mean:.1f}W E_total={total_j:.1f}J E_marg={marginal_j:.1f}J"
    )


if __name__ == "__main__":
    main()
