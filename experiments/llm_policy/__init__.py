"""LLM-as-Policy experiment: a language model as the configuration-selection policy.

Responsibility and boundaries: this package adapts a local instruction-tuned LLM to the
frozen V1 ``Policy`` protocol. The model sees exactly what every other policy sees — the
contract, the frozen ``RuntimeState``, and the allowed configurations with their public
profiles — serialized into a deterministic prompt. It never sees ground truth, the
network trace, or the simulator. Heavy dependencies (torch, transformers) live here and
only here, imported lazily inside the transformers backend; the core package and the CI
tests use an injected fake backend and never load a model.
"""
