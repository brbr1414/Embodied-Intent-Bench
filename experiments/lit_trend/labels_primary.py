"""Manual relevance labels for the v1 PRIMARY census (all 78 works), 2026-08-18.

Labeled by reading title + abstract excerpt against the definition: the work is
about runtime/adaptive selection or switching among AI models or inference
configurations on resource-constrained / edge platforms. R = relevant,
B = borderline (adjacent: adaptive execution, deployment-time selection, or
selection outside the strict edge-inference frame), I = irrelevant.
"""
# ruff: noqa: E501  (annotation data: one-sentence reasons exceed the line limit)

LABELS = {
    "W2204230881": (
        "I",
        "Bayesian statistical model selection for urban mobility, not AI-system runtime selection",
    ),
    "W2514509635": ("I", "control-theoretic model selection for synthetic gene circuits"),
    "W2794478957": (
        "B",
        "on-device incremental DNN adaptation for IoT, but not selection among models/configs",
    ),
    "W2950576443": (
        "B",
        "DRL selects NB-IoT radio configurations at runtime - configured artifact is the network, not the AI system",
    ),
    "W2957242392": ("I", "railway route setting and rescheduling; phrase match incidental"),
    "W2905601756": ("B", "same NB-IoT radio-configuration selection line as W2950576443"),
    "W2948277279": (
        "I",
        "EEG sleep-spindle detection with generative models; no runtime selection",
    ),
    "W2979811226": (
        "B",
        "hardware/configuration choice for DNN inference deployment, but static not runtime",
    ),
    "W7171228761": (
        "B",
        "energy-efficient serverless ML inference pipelines; explicit runtime selection unclear",
    ),
    "W3004633656": ("R", "canonical adaptive model selection for embedded DNN inference"),
    "W3013634724": (
        "B",
        "survey of self-aware/self-adaptive NN systems; adjacent to runtime adaptation",
    ),
    "W3081575380": (
        "B",
        "online Bayesian fusion of predictors on mobile; fusion not configuration selection",
    ),
    "W3096084597": ("R", "runtime neural-configuration adaptation for edge AR video analysis"),
    "W3113668354": (
        "B",
        "cloud-to-edge model transfer architecture; no explicit runtime selection",
    ),
    "W3134268474": ("I", "airline ancillary pricing field experiment; phrase match incidental"),
    "W3137964896": ("I", "authorship attribution; statistical model stacking"),
    "W3173385564": ("R", "contention-aware adaptive model selection on embedded vision systems"),
    "W3173929316": (
        "R",
        "FPGA time-division multiplexing accounting for CNN model-switch cost at the edge",
    ),
    "W3191769047": (
        "R",
        "contextual-bandit adaptive model selection across hierarchical edge for IoT anomaly detection",
    ),
    "W3201628041": (
        "R",
        "runtime random precision switching for DNN inference on IoT accelerators",
    ),
    "W3207464805": (
        "R",
        "content-aware runtime configuration adaptation for edge video streaming/analysis",
    ),
    "W3211797015": ("I", "vehicle-velocity forecasting dataset; phrase match incidental"),
    "W4220807708": ("R", "contention-graded adaptive model selection on embedded vision systems"),
    "W4280651189": (
        "B",
        "adaptive dataflow CNN acceleration on FPGA; runtime vs synthesis-time configuration unclear",
    ),
    "W4284686519": ("I", "mobile-crane configuration selection in construction"),
    "W4285197124": (
        "R",
        "online configuration adaptation + model selection + provisioning for edge DNN serving",
    ),
    "W4285815180": ("R", "multi-model storage and on-device switching on microcontrollers"),
    "W4291700252": ("R", "methodology critique of premodel-based adaptive CNN model selection"),
    "W4310398573": ("B", "distributed/partitioned CNN edge inference; partitioning not selection"),
    "W4311224386": ("I", "static multi-objective DNN pruning; no runtime selection"),
    "W4318829126": (
        "B",
        "title suggests adaptive mixed-precision execution but no abstract available",
    ),
    "W4319952106": (
        "B",
        "robustness/security thesis; includes precision-switching chapter but broader topic",
    ),
    "W4361019767": ("R", "content-adaptive model selection for video super-resolution inference"),
    "W7203517558": (
        "B",
        "algorithm-hardware co-design thesis; runtime precision switching is one chapter",
    ),
    "W4317584133": ("I", "aerospace conceptual-design configuration selection"),
    "W4386952618": (
        "B",
        "offline design-space search over embedded hardware configurations, not runtime",
    ),
    "W4386987491": ("R", "runtime switching among pruned DNN variants on edge"),
    "W4391468567": ("R", "dynamic DNN model switching for UAV-swarm edge intelligence"),
    "W4393186512": (
        "R",
        "runtime light/heavy teacher-student model adaptation for MEC video inference",
    ),
    "W4395685061": (
        "R",
        "edge system dynamically selects between on-device model and cloud foundation model",
    ),
    "W4392646481": ("B", "ViT overlay processor for edge; no abstract, selection unclear"),
    "W4392885553": (
        "R",
        "adaptive model selection and partition for edge intelligence under dynamic resources",
    ),
    "W4392910469": (
        "R",
        "edge-assisted model switching for video recognition over variable networks",
    ),
    "W4393140900": ("I", "PIM-assisted hyperdimensional-computing acceleration; no selection"),
    "W4402985646": ("I", "offline comparison of diffusion-model fine-tuning techniques"),
    "W4403882605": (
        "B",
        "learned LLM embeddings for model routing, but not an edge/resource-constrained setting",
    ),
    "W4403968670": ("I", "LoRaWAN localization parameter tuning"),
    "W4404177704": (
        "B",
        "on-device SR with dynamic algorithm/compiler co-design; no abstract to confirm selection",
    ),
    "W4412610540": (
        "R",
        "adaptive configuration selection for multi-model edge inference pipelines",
    ),
    "W7126565908": ("R", "dynamic model selection by content and resolution for video analytics"),
    "W4409129983": ("R", "adaptive DNN model switching on edge driven by resource benchmarks"),
    "W4410029827": (
        "R",
        "adaptive model switching for collaborative multi-CNN inference in UAV swarms",
    ),
    "W4410087102": (
        "B",
        "SNR-driven adaptive fusion of CNN-GRU under compute constraints; fusion not selection",
    ),
    "W4410216893": ("R", "FPGA system with runtime dynamic model selection"),
    "W4410906901": ("I", "offline local-LLM RAG system; no runtime selection"),
    "W4411599698": (
        "R",
        "nested-precision quantized variants switched on-device for resource adaptation",
    ),
    "W4413068226": (
        "B",
        "adaptive lightweight-vs-heavy security models on sensor nodes; mechanism unclear",
    ),
    "W4413288901": ("I", "broad AI-for-smart-cities analytics; no evidence of selection"),
    "W4413553381": (
        "R",
        "joint container management and model selection for heterogeneous edge computing",
    ),
    "W4414165967": (
        "R",
        "dynamic quality-latency aware LLM inference routing in wireless edge-device networks",
    ),
    "W4414447071": (
        "R",
        "energy-adaptive perception configuration via policy learning for autonomous driving",
    ),
    "W4414603846": ("B", "adaptive 6G edge security framework; model selection incidental"),
    "W4414898763": ("R", "online AI model selection and placement for carbon-aware edge inference"),
    "W4414956594": (
        "B",
        "edge health monitoring with an adaptive CNN-LSTM; single model, no selection",
    ),
    "W4415068610": (
        "R",
        "black-box edge AI model selection with conformal latency/accuracy guarantees",
    ),
    "W4415624979": ("B", "on-device deep-learning energy optimization; selection aspect unclear"),
    "W4416034518": ("R", "adaptive LLM routing under budget constraints"),
    "W4416271107": (
        "I",
        "SR accelerator; 'edge selective' refers to image edges, not model selection",
    ),
    "W4417119112": (
        "R",
        "characterization of multi-LLM routing and hierarchical inference for efficiency",
    ),
    "W4417282405": (
        "B",
        "user-chosen accuracy-energy configurations on TinyML, not autonomous runtime selection",
    ),
    "W4417402941": ("R", "adaptive model switching for inference serving across IoT-edge-cloud"),
    "W7128623872": ("R", "resource-aware search over models for on-device NLP inference"),
    "W7131838354": ("R", "dynamic model-switching pipeline on highly constrained edge devices"),
    "W7134895093": ("I", "efficient single-model FPGA gesture recognition; no selection"),
    "W7138267987": (
        "R",
        "temperature-adaptive dynamic switching of pre-pruned models on satellite edge",
    ),
    "W7138935118": (
        "I",
        "federated fine-tuning of foundation models; training-side, no runtime selection",
    ),
    "W7140283802": ("R", "dynamic model selection for energy-efficient healthcare AI"),
    "W7141920270": ("I", "UAV hardware configuration design for environmental monitoring"),
    "W4415974678": (
        "B",
        "runtime FPGA partial reconfiguration of DSP configurations; AI-model selection unclear",
    ),
    "W4416151651": ("R", "dynamic LLM selection with edge caching for humanoid robots"),
    "W7118093888": (
        "B",
        "cost-aware multi-model LLM routing, but enterprise CRM rather than edge/resource-constrained",
    ),
    "W7123640040": (
        "B",
        "edge model serving under data drift; checkpoint management rather than configuration selection",
    ),
}
