"""Manual relevance labels for the BROAD validation sample (2026-08-18).

Judged against the Broad Adaptive Execution intent: runtime-adaptive execution or
adaptation of AI inference under resource constraints (partitioning, early exit,
dynamic networks, adaptive serving included by design).
"""
# ruff: noqa: E501  (annotation data: one-sentence reasons exceed the line limit)

LABELS = {
    "W2962752334": ("R", "motion-adaptive inference for real-time embedded object detection"),
    "W2910908473": (
        "R",
        "adaptive collaborative inference of distributed DL models on IoT devices",
    ),
    "W4302583374": (
        "R",
        "same Fast YOLO work as W2962752334 (arXiv version; residual duplicate, noted in limitations)",
    ),
    "W2927152095": ("R", "context-aware dynamic network blocks for resource-constrained inference"),
    "W1491941570": ("I", "guest editorial, not a research paper"),
    "W2920031528": ("R", "runtime adaptive DNN partitioning for edge inference acceleration"),
    "W3009921999": ("R", "privacy-aware adaptive DNN partitioning for edge computing"),
    "W3003782683": (
        "R",
        "adaptive DNN execution for video surveillance in distributed edge clouds",
    ),
    "W2768646057": ("R", "dynamic DNN design for workload allocation at the edge"),
    "W3198907060": ("R", "adaptive DNN partitioning between mobile terminal and edge server"),
    "W3199427948": ("I", "deep-learning systems security thesis; no adaptive execution"),
    "W3047467872": ("R", "dimension-adaptive neural inference for sensor streams on wearables"),
    "W3160211771": ("R", "dynamic multi-path network execution for resource-limited devices"),
    "W3203454415": ("I", "memristor state-drift mitigation; device physics"),
    "W3200878837": ("R", "adaptive inference via CNN pruning and HLS versioning on multi-FPGAs"),
    "W4287110835": ("R", "multi-exit vision transformer for dynamic inference"),
    "W3200236657": ("R", "context-aware adaptive DNN partitioning on end devices"),
    "W3204165788": ("R", "dynamic DNN decomposition for synergistic mobile-edge inference"),
    "W3124022277": (
        "B",
        "rules-engine offloading decisions in edge building systems; loosely adaptive",
    ),
    "W4392944897": ("B", "adaptive real-time industrial edge DL ecosystem; vague mechanism"),
    "W4387968598": ("R", "platform-agnostic adaptive edge-cloud DNN partitioning"),
    "W4386880906": ("B", "latency-critical DL inference serving; adaptivity unclear"),
    "W4390585142": ("R", "adaptive neural network execution for edge intelligence"),
    "W4377001532": ("R", "coupled model compression and dynamic inference for edge AI"),
    "W4386765240": ("R", "energy-efficient networks with runtime power management"),
    "W4366967707": (
        "R",
        "wireless-channel-adaptive DNN split inference on constrained edge devices",
    ),
    "W4381714077": ("R", "adaptive DNN surgery with on-demand edge resources"),
    "W4389082012": ("I", "analog in-memory computing toolkit; no adaptive execution"),
    "W4366833205": ("R", "input-difficulty-aware dynamic DNN pruning"),
    "W4401568158": ("R", "fluid dynamic DNNs for adaptive distributed edge inference"),
    "W4416430310": ("B", "generic adaptive deep-learning-on-edge architecture paper"),
    "W4399991074": ("I", "efficient neuromorphic keyword spotting; static implementation"),
    "W4402449925": ("R", "adaptive DNN splitting in multi-UAV sensing-communication-computation"),
    "W4399121259": ("R", "adaptive DNN inference on memory-constrained edge devices"),
    "W4403196715": ("R", "survey of early-exit dynamic inference"),
    "W4394904485": ("I", "memristive synapse device improvement"),
    "W4404388593": ("R", "energy-aware dynamic neural inference"),
    "W4404034550": ("R", "spatial-sparsity-driven dynamic DNN inference on devices"),
    "W4402834073": ("R", "dynamic batching and early exit for timely edge inference"),
    "W4411725088": (
        "R",
        "image-difficulty-driven runtime-adaptive DNN inference on embedded devices",
    ),
    "W4414169753": ("R", "joint adaptive split inference and bandwidth allocation at the edge"),
    "W4415686238": ("B", "review of resource-constrained embedded vision; adaptivity partial"),
    "W4408017328": ("B", "edge-assisted multi-task perception system; adaptive execution partial"),
    "W4412164080": (
        "R",
        "joint DNN deployment, selection, and configuration for edge inference services",
    ),
    "W4413442912": ("R", "channel-adaptive near-sensor accelerator for dynamic inference"),
    "W4411607398": ("B", "edge SoC claiming task-adaptive inference; architecture paper"),
    "W4416158214": ("R", "adaptive LLM chunking for inference on automotive edge devices"),
    "W4416613003": ("I", "cross-validation statistical theory; false positive"),
    "W4410583443": (
        "R",
        "joint DNN partition and thread allocation for energy-harvesting MEC inference",
    ),
}
