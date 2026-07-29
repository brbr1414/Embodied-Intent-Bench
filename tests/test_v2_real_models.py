"""V2.1 real-model executors: unit tests on an injected fake backend, plus opt-in
tests that load the actual pretrained weights.

The default suite never imports torch, never downloads weights, and needs no GPU: the
executor's backend seam accepts any object with ``infer(rgb) -> (prob_map, latency)``
and an ``info`` record, so everything from config validation to full-mission dispatch is
exercised with a numpy fake. The ``real_models``-marked tests at the bottom construct
the genuine torchvision backends and are deselected by default
(``addopts = -m 'not real_models'``); run them with ``pytest -m real_models``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError  # noqa: E402
from aerointentbench.v2.real_models import (  # noqa: E402
    BackendInfo,
    TorchSemanticSegmentationExecutor,
    check_real_model_availability,
)
from aerointentbench.v2.runner import MissionRunner, build_policy  # noqa: E402
from aerointentbench.v2.scenario import load_scenario  # noqa: E402
from tests.test_v2_visual_loop import scenario_payload, write_world_png  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REAL_SCENARIO = REPO / "data" / "v2_scenarios" / "demo_img1_real_models.json"


# --- fixtures --------------------------------------------------------------------------------


def torch_config(config_id: str = "local_light_real", **overrides) -> dict:
    config = {
        "config_id": config_id,
        "model_strategy_id": "FAKE_SEG",
        "kind": "torch_semantic_segmentation",
        "mission_latency_s": 0.15,
        "energy_j_per_call": 2.0,
        "communication_mb_per_call": 0.0,
        "quality_tier": "low",
        "parameters": {
            "model_id": "lraspp_mobilenet_v3_large",
            "weights_id": "official_default",
            "task_class": "person",
            "input_width_px": 64,
            "input_height_px": 48,
            "probability_threshold": 0.5,
            "device": "auto",
            "dtype": "float32",
            "latency_mode": "measured",
            "warmup_runs": 0,
            "min_component_px": 0,
        },
    }
    config["parameters"].update(overrides.pop("parameters", {}))
    config.update(overrides)
    return config


def real_scenario_path(tmp_path: Path, *configs: dict, fallback: str = "local_light_real") -> Path:
    world = write_world_png(tmp_path / "world.png")
    payload = scenario_payload(world)
    payload["executor_configs"] = list(configs) or [
        torch_config("local_light_real", quality_tier="low"),
        torch_config(
            "local_strong_real",
            quality_tier="high",
            parameters={"input_width_px": 96, "input_height_px": 72},
        ),
    ]
    payload["simulation"]["fallback_config_id"] = fallback
    path = tmp_path / "real_scenario.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeBackend:
    """A deterministic stand-in with the real backend's surface. Not a neural network."""

    def __init__(self, *, hot_threshold: int = 180, latency_s: float = 0.001) -> None:
        self.calls = 0
        self._hot = hot_threshold
        self._latency = latency_s
        self.info = BackendInfo(
            framework="fake",
            framework_version="0",
            model_id="fake_seg",
            weights_id="none",
            weights_identifier="FakeWeights.NONE",
            target_class="person",
            target_class_index=15,
            category_count=21,
            device_requested="auto",
            device_used="cpu",
            dtype_requested="float32",
            dtype_used="float32",
            input_width_px=64,
            input_height_px=48,
            preprocessing_id="fake",
            model_load_s=0.0,
            warmup_runs=0,
        )

    def infer(self, rgb_input):
        self.calls += 1
        # "Probability" = red-channel intensity: content-dependent, GT-independent.
        prob = (rgb_input[..., 0].astype(np.float32) >= self._hot).astype(np.float32)
        return prob, self._latency


def executor_with_fake(config: dict | None = None, backend: FakeBackend | None = None):
    from aerointentbench.v2.scenario import ExecutorConfigSpec

    raw = config or torch_config()
    spec = ExecutorConfigSpec(
        config_id=raw["config_id"],
        model_strategy_id=raw["model_strategy_id"],
        kind=raw["kind"],
        mission_latency_s=raw["mission_latency_s"],
        energy_j_per_call=raw["energy_j_per_call"],
        communication_mb_per_call=raw["communication_mb_per_call"],
        quality_tier=raw["quality_tier"],
        parameters=raw["parameters"],
    )
    fake = backend or FakeBackend()
    return TorchSemanticSegmentationExecutor(spec, backend=fake), fake


# --- config validation -----------------------------------------------------------------------


def test_a_valid_real_model_scenario_loads(tmp_path: Path) -> None:
    scenario = load_scenario(real_scenario_path(tmp_path))
    kinds = {s.kind for s in scenario.executor_configs}
    assert kinds == {"torch_semantic_segmentation"}


def test_missing_required_model_parameters_fail(tmp_path: Path) -> None:
    broken = torch_config()
    del broken["parameters"]["model_id"]
    with pytest.raises(SchemaValidationError, match="missing \\['model_id'\\]"):
        load_scenario(real_scenario_path(tmp_path, broken, torch_config("other")))


def test_a_bad_latency_mode_fails(tmp_path: Path) -> None:
    broken = torch_config(parameters={"latency_mode": "guessed"})
    with pytest.raises(SchemaValidationError, match="latency_mode"):
        load_scenario(real_scenario_path(tmp_path, broken, torch_config("other")))


def test_a_bad_dtype_fails(tmp_path: Path) -> None:
    broken = torch_config(parameters={"dtype": "int8"})
    with pytest.raises(SchemaValidationError, match="dtype"):
        load_scenario(real_scenario_path(tmp_path, broken, torch_config("other")))


def test_a_bad_threshold_fails(tmp_path: Path) -> None:
    broken = torch_config(parameters={"probability_threshold": 1.5})
    with pytest.raises(SchemaValidationError, match="probability_threshold"):
        load_scenario(real_scenario_path(tmp_path, broken, torch_config("other")))


def test_shipped_real_scenario_validates() -> None:
    scenario = load_scenario(REAL_SCENARIO)
    assert scenario.config_ids == ("local_light_real", "local_strong_real")
    for spec in scenario.executor_configs:
        assert spec.parameters["latency_mode"] == "measured"


def test_shipped_hard_tradeoff_scenario_pins_its_design() -> None:
    """The hard scenario's trade-off structure is deliberate; pin what makes it work.

    The mission outcomes themselves need local assets + torch and live under
    ``results/v2_hard`` (see docs/v2_design.md §10.5); this pins the machine-independent
    design invariants so a casual edit cannot silently defuse the trade-off.
    """
    scenario = load_scenario(REPO / "data" / "v2_scenarios" / "demo_img1_hard_tradeoff.json")
    assert scenario.config_ids == ("local_light_real", "local_strong_real")
    light, strong = scenario.executor_configs
    interval = scenario.simulation.observation_interval_s
    # Deterministic missions: the clock runs on configured latency, never wall-clock.
    assert {s.parameters["latency_mode"] for s in scenario.executor_configs} == {"configured"}
    # Light processes every capture; strong skips every other one (the coverage cost)...
    assert light.mission_latency_s < interval < strong.mission_latency_s
    # ...but stays under the rule-based latency ceiling (1.5 intervals), so the adaptive
    # policy genuinely prefers it until battery pressure bites.
    assert strong.mission_latency_s <= 1.5 * interval
    # Battery pressure is the adaptive switch driver: strong's stress-configured energy
    # must dominate flight power at its 2-capture cadence.
    assert strong.energy_j_per_call / (2 * interval) > scenario.drone.flight_power_w
    assert len(scenario.targets) == 8
    assert all(obj.render_mode == "image_asset" for obj in scenario.targets)
    assert scenario.simulation.fallback_config_id == "local_light_real"
    assert scenario.contract.quality_threshold == 0.7
    assert scenario.evaluation_purpose == "controlled_observability"


# --- executor behaviour on the fake backend --------------------------------------------------


def test_prediction_depends_on_rgb_and_matches_observation_shape() -> None:
    executor, fake = executor_with_fake()
    blank = np.full((48, 64, 3), 100, dtype=np.uint8)
    marked = blank.copy()
    marked[10:20, 10:30, 0] = 255  # hot red region -> "person" probability 1.0
    empty = executor.run(blank)
    found = executor.run(marked)
    assert empty.prediction_mask.shape == (48, 64) == found.prediction_mask.shape
    assert not empty.prediction_mask.any()
    assert found.prediction_mask.any()
    assert fake.calls == 2


def test_mask_is_resized_back_to_the_observation_resolution() -> None:
    executor, _ = executor_with_fake()  # model input 64x48
    big = np.full((192, 256, 3), 100, dtype=np.uint8)
    big[40:80, 40:120, 0] = 255
    result = executor.run(big)
    assert result.prediction_mask.shape == (192, 256)  # observation, not model input
    assert result.prediction_mask.any()
    assert result.diagnostics["observation_width_px"] == 256
    assert result.diagnostics["input_width_px"] == 64


def test_threshold_is_applied() -> None:
    # Fake emits probability 1.0 only where red >= 180; a threshold above 1.0 is invalid,
    # so instead verify via a backend whose hot region is graded.
    class Graded(FakeBackend):
        def infer(self, rgb_input):
            self.calls += 1
            return rgb_input[..., 0].astype(np.float32) / 255.0, 0.001

    low = torch_config(parameters={"probability_threshold": 0.3})
    high = torch_config(parameters={"probability_threshold": 0.9})
    rgb = np.full((48, 64, 3), 0, dtype=np.uint8)
    rgb[:, :, 0] = 128  # probability 0.502 everywhere
    executor_low, _ = executor_with_fake(low, Graded())
    executor_high, _ = executor_with_fake(high, Graded())
    assert executor_low.run(rgb).prediction_mask.all()
    assert not executor_high.run(rgb).prediction_mask.any()


def test_min_component_postprocessing_is_config_driven() -> None:
    config = torch_config(parameters={"min_component_px": 50})
    executor, _ = executor_with_fake(config)
    rgb = np.full((48, 64, 3), 100, dtype=np.uint8)
    rgb[10:12, 10:12, 0] = 255  # 4 px at model res: below min_component_px
    result = executor.run(rgb)
    assert not result.prediction_mask.any()
    assert result.diagnostics["postprocessing"] == {"min_component_px": 50}


def test_measured_latency_mode_drives_the_mission_clock() -> None:
    executor, _ = executor_with_fake()
    result = executor.run(np.full((48, 64, 3), 100, dtype=np.uint8))
    assert result.mission_latency_s == result.measured_wall_clock_s > 0.0
    assert "measured" in result.measurement_provenance["mission_latency_s"]
    assert result.diagnostics["latency_mode"] == "measured"


def test_configured_latency_mode_keeps_the_profile_value() -> None:
    config = torch_config(parameters={"latency_mode": "configured"})
    executor, _ = executor_with_fake(config)
    result = executor.run(np.full((48, 64, 3), 100, dtype=np.uint8))
    assert result.mission_latency_s == 0.15  # the configured profile value
    assert result.measured_wall_clock_s != result.mission_latency_s
    assert "simulated" in result.measurement_provenance["mission_latency_s"]
    assert "diagnostic" in result.measurement_provenance["mission_latency_s"]


def test_result_provenance_is_complete() -> None:
    executor, _ = executor_with_fake()
    d = executor.run(np.full((48, 64, 3), 100, dtype=np.uint8)).diagnostics
    for key in (
        "executor_kind",
        "framework",
        "framework_version",
        "model_id",
        "weights_id",
        "weights_identifier",
        "target_class",
        "target_class_index",
        "device_requested",
        "device_used",
        "dtype_requested",
        "dtype_used",
        "preprocessing_id",
        "model_load_s",
        "warmup_runs",
        "probability_threshold",
        "postprocessing",
        "latency_mode",
        "model_forward_latency_s",
        "end_to_end_executor_latency_s",
        "predicted_positive_px",
    ):
        assert key in d, key
    assert (
        "simulated"
        in executor.run(np.full((48, 64, 3), 100, dtype=np.uint8)).measurement_provenance[
            "energy_j"
        ]
    )


def test_executor_surface_is_rgb_only() -> None:
    import inspect

    import aerointentbench.v2.real_models as module

    executor, _ = executor_with_fake()
    assert list(inspect.signature(executor.run).parameters) == ["rgb"]
    source = inspect.getsource(module)
    for forbidden in ("semantic_gt", "instance_gt", "visible_target_ids", "ObjectLayer"):
        assert forbidden not in source, f"real_models must not reference {forbidden}"


# --- registry dispatch, caching, and the mission loop ----------------------------------------


def test_missing_torch_produces_an_actionable_error(monkeypatch, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name in ("torch", "torchvision") or name.startswith(("torch.", "torchvision.")):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    from aerointentbench.v2.real_models import TorchSegmentationBackend

    with pytest.raises(SchemaValidationError, match="v2-real-models"):
        TorchSegmentationBackend(
            model_id="lraspp_mobilenet_v3_large",
            weights_id="official_default",
            target_class="person",
            device="auto",
            dtype="float32",
            input_width_px=64,
            input_height_px=48,
        )


def test_backends_are_cached_per_strategy(monkeypatch) -> None:
    from aerointentbench.v2 import real_models

    created: list[str] = []

    class Counting(FakeBackend):
        pass

    def factory(**kwargs):
        created.append(kwargs["model_id"])
        return Counting()

    monkeypatch.setattr(real_models, "TorchSegmentationBackend", lambda **kw: factory(**kw))
    real_models.clear_backend_cache()
    params = dict(torch_config()["parameters"])
    one = real_models.backend_for(params)
    two = real_models.backend_for(dict(params))  # identical strategy -> cached
    assert one is two and created == ["lraspp_mobilenet_v3_large"]
    other = real_models.backend_for({**params, "input_width_px": 128})
    assert other is not one and len(created) == 2  # a different strategy loads its own
    real_models.clear_backend_cache()


def test_policies_resolve_real_configs(tmp_path: Path) -> None:
    scenario = load_scenario(real_scenario_path(tmp_path))
    assert build_policy("always_light_real", scenario).config_id == "local_light_real"
    assert build_policy("always_strong_real", scenario).config_id == "local_strong_real"
    assert build_policy("local_strong_real", scenario).config_id == "local_strong_real"


def test_always_real_policies_fail_without_real_configs(tmp_path: Path) -> None:
    world = write_world_png(tmp_path / "w.png")
    payload = scenario_payload(world)  # heuristic-only scenario
    path = tmp_path / "heuristic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="always_light_real"):
        build_policy("always_light_real", load_scenario(path))


def test_the_mission_loop_dispatches_the_selected_real_executor(tmp_path: Path) -> None:
    """End to end with fakes: policy -> config_id -> injected backend -> closed loop."""
    import aerointentbench.v2.real_models as rm

    scenario = load_scenario(real_scenario_path(tmp_path))
    fakes: dict[str, FakeBackend] = {}
    original = rm.backend_for

    def fake_factory(params):
        fake = FakeBackend()
        fakes[str(params["input_width_px"])] = fake
        return fake

    try:
        rm.clear_backend_cache()
        rm.backend_for = fake_factory
        mission = MissionRunner(scenario, "always_strong_real")
        result = mission.run()
    finally:
        rm.backend_for = original

    assert set(result.config_selection_history) == {"local_strong_real"}
    assert result.processed_observation_count > 0
    strong_fake = fakes["96"]
    assert strong_fake.calls >= result.processed_observation_count
    first = result.observations[0]
    assert first.execution["diagnostics"]["executor_kind"] == "torch_semantic_segmentation"
    assert first.execution["diagnostics"]["latency_mode"] == "measured"


# --- opt-in: the actual pretrained models ----------------------------------------------------


@pytest.mark.real_models
def test_actual_pretrained_backends_run_a_forward_pass() -> None:
    torch = pytest.importorskip("torch", reason="needs the [v2-real-models] extras")
    del torch
    from aerointentbench.v2.real_models import backend_for, clear_backend_cache

    clear_backend_cache()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (192, 256, 3), dtype=np.uint8)
    for model_id, w, h in (
        ("lraspp_mobilenet_v3_large", 256, 192),
        ("deeplabv3_resnet50", 256, 192),
    ):
        backend = backend_for(
            {
                "model_id": model_id,
                "weights_id": "official_default",
                "task_class": "person",
                "device": "auto",
                "dtype": "float32",
                "input_width_px": w,
                "input_height_px": h,
                "warmup_runs": 1,
            }
        )
        resized = np.array(rgb[:h, :w])
        prob, forward_s = backend.infer(resized)
        assert prob.shape == (h, w)
        assert prob.dtype == np.float32
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
        assert forward_s > 0.0
        assert backend.info.target_class == "person"
        assert backend.info.target_class_index > 0  # resolved from weight metadata
    clear_backend_cache()


@pytest.mark.real_models
def test_check_real_models_reports_ok_for_the_shipped_scenario() -> None:
    pytest.importorskip("torchvision", reason="needs the [v2-real-models] extras")
    scenario = load_scenario(REAL_SCENARIO)
    reports = check_real_model_availability(
        [dict(s.parameters) for s in scenario.executor_configs], load=False
    )
    assert all(r["status"] == "ok" for r in reports), reports
    assert all(r["target_class_index"] is not None for r in reports)
