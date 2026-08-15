"""Local instruction-model backend for the LLM policy (torch/transformers live HERE).

``choice``: each option id is scored as a continuation of the prompt — the summed
log-probability of the option's tokens given the prompt (length-normalized), computed in
one forward pass per option. Argmax is deterministic for a fixed model and device; no
sampling anywhere.

``generate``: greedy decoding, short budget, chat template — the mode that shows the
model's unconstrained behaviour, parse failures included.

The model id, dtype, and device are recorded so a result can name exactly what decided.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TransformersBackend"]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class TransformersBackend:
    """Greedy, sampling-free backend over a local Hugging Face causal LM."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, max_new_tokens: int = 24) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
            .to(self.device)
            .eval()
        )

    def describe(self) -> dict[str, str]:
        return {"model_id": self.model_id, "device": self.device, "dtype": "float32"}

    # -- choice mode --------------------------------------------------------------------

    def choose(self, prompt: str, options: list[str]) -> str:
        torch = self._torch
        prompt_ids = self._chat_ids(prompt)
        scores: list[float] = []
        with torch.no_grad():
            for option in options:
                option_ids = self._tokenizer(" " + option, add_special_tokens=False).input_ids
                ids = torch.tensor([prompt_ids + option_ids], device=self.device)
                logits = self._model(ids).logits[0]
                # Log-prob of each option token given everything before it.
                total = 0.0
                for position, token_id in enumerate(option_ids):
                    step = len(prompt_ids) + position - 1
                    log_probs = torch.log_softmax(logits[step], dim=-1)
                    total += float(log_probs[token_id])
                scores.append(total / len(option_ids))
        best = max(range(len(options)), key=lambda i: (scores[i], -i))
        return options[best]

    # -- generate mode ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        torch = self._torch
        ids = torch.tensor([self._chat_ids(prompt)], device=self.device)
        with torch.no_grad():
            output = self._model.generate(
                ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(output[0][ids.shape[1] :], skip_special_tokens=True)

    # -- helpers ------------------------------------------------------------------------

    def _chat_ids(self, prompt: str) -> list[int]:
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._tokenizer(text, add_special_tokens=False).input_ids


def default_cache_note() -> str:
    hub = Path.home() / ".cache" / "huggingface"
    return f"weights resolve from {hub} (downloaded once by the intent-compiler work)"
