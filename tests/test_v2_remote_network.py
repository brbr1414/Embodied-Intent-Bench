"""V3 P1: simulated remote inference, dynamic network, and the fifth (privacy) constraint.

Everything runs CI-safe on the tiny generated PNG world with the heuristic executors —
no Torch, no assets, no server, no GPU, no internet. The simulated transport and
backend are deterministic, so every assertion here is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import SchemaValidationError  # noqa: E402
from aerointentbench.v2.network import (  # noqa: E402
    NetworkRegime,
    V2NetworkModel,
    constant_network_model,
)
from aerointentbench.v2.remote import (  # noqa: E402
    PROTOCOL_VERSION,
    REMOTE_PARAMETER_DEFAULTS,
    ExecutionContext,
    InferenceRequest,
    RemoteStatus,
    SimulatedRemoteExecutor,
    SimulatedTransport,
    build_remote_executor,
)
from aerointentbench.v2.runner import MissionRunner, run_mission  # noqa: E402
from aerointentbench.v2.scenario import ExecutorConfigSpec, load_scenario  # noqa: E402
from tests.test_v2_visual_loop import scenario_payload, write_world_png  # noqa: E402

# --- fixtures --------------------------------------------------------------------------------

GOOD = NetworkRegime("5g_good", 0.0, 100.0, 200.0, 20.0, 0.0)
DEGRADED = NetworkRegime("lte_degraded", 10.0, 2.0, 5.0, 150.0, 0.05)
LOSSY = NetworkRegime("lossy", 14.0, 50.0, 100.0, 40.0, 0.6)
DISCONNECTED = NetworkRegime("disconnected", 18.0, 0.0, 0.0, 0.0, 1.0)
RECOVERED = NetworkRegime("recovered", 24.0, 50.0, 100.0, 30.0, 0.0)

REMOTE_CONFIG = {
    "config_id": "REMOTE_STRONG",
    "model_strategy_id": "T_REMOTE_STRONG",
    "kind": "simulated_remote",
    "mission_latency_s": 0.9,
    "energy_j_per_call": 0.0,
    "communication_mb_per_call": 2.0,
    "quality_tier": "high",
    "parameters": {
        "backend_kind": "slow_strong",
        "remote_compute_s": 0.3,
        "timeout_s": 3.0,
        "fallback_config_id": "FAST",
    },
}
TRACE = [
    {
        "regime_id": "5g_good",
        "start_s": 0.0,
        "uplink_mbps": 100.0,
        "downlink_mbps": 200.0,
        "rtt_ms": 20.0,
        "packet_loss_frac": 0.0,
    },
    {
        "regime_id": "lte_degraded",
        "start_s": 10.0,
        "uplink_mbps": 2.0,
        "downlink_mbps": 5.0,
        "rtt_ms": 150.0,
        "packet_loss_frac": 0.05,
    },
    {
        "regime_id": "disconnected",
        "start_s": 18.0,
        "uplink_mbps": 0.0,
        "downlink_mbps": 0.0,
        "rtt_ms": 0.0,
        "packet_loss_frac": 1.0,
    },
    {
        "regime_id": "recovered",
        "start_s": 24.0,
        "uplink_mbps": 50.0,
        "downlink_mbps": 100.0,
        "rtt_ms": 30.0,
        "packet_loss_frac": 0.0,
    },
]


@pytest.fixture
def make_scenario(tmp_path: Path):
    world = write_world_png(tmp_path / "world.png")

    def _make(**overrides) -> Path:
        payload = scenario_payload(world)
        payload["executor_configs"].append(json.loads(json.dumps(REMOTE_CONFIG)))
        payload["simulation"]["network_trace"] = json.loads(json.dumps(TRACE))
        for dotted, value in overrides.items():
            node = payload
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _make


class _FixedBackend:
    """A deterministic stand-in server model for transport unit tests."""

    model_strategy_id = "FIXED"

    def infer(self, request, rgb):
        return np.ones(rgb.shape[:2], dtype=bool)


class _RaisingBackend:
    model_strategy_id = "RAISING"

    def infer(self, request, rgb):
        raise RuntimeError("server exploded")


def transport(backend=None, **param_overrides) -> SimulatedTransport:
    params = {
        **{k: float(v) for k, v in REMOTE_PARAMETER_DEFAULTS.items()},
        "remote_compute_s": 0.3,
        **param_overrides,
    }
    return SimulatedTransport(params, backend or _FixedBackend())


def request(payload_mb: float = 2.0) -> InferenceRequest:
    return InferenceRequest(
        request_id="S/obs000001",
        scenario_id="S",
        observation_id=1,
        capture_time_s=1.0,
        deadline_time_s=60.0,
        requested_config_id="REMOTE_STRONG",
        payload_kind="raw_rgb",
        payload_shape=(48, 64, 3),
        payload_mb=payload_mb,
        privacy_level="remote_allowed",
    )


RGB = np.full((48, 64, 3), 120, dtype=np.uint8)


# --- latency semantics -----------------------------------------------------------------------


def test_latency_formula_charges_each_stage_and_rtt_exactly_once() -> None:
    result = transport().execute(request(payload_mb=2.0), RGB, GOOD, timeout_s=60.0)
    assert result.status is RemoteStatus.SUCCESS
    upload_s = 2.0 * 8.0 / 100.0
    download_s = 0.05 * 8.0 / 200.0
    expected = 0.02 + upload_s + 0.020 + 0.0 + 0.3 + download_s + 0.01
    assert result.elapsed_s == pytest.approx(expected)
    stages = result.stages_s
    assert stages["upload_s"] == pytest.approx(upload_s)
    assert stages["download_s"] == pytest.approx(download_s)
    assert stages["rtt_s"] == pytest.approx(0.020)  # once, never per leg
    assert result.elapsed_s == pytest.approx(sum(stages.values()))


def test_zero_bandwidth_is_unreachable_with_nothing_transmitted() -> None:
    result = transport().execute(request(), RGB, DISCONNECTED, timeout_s=60.0)
    assert result.status is RemoteStatus.UNREACHABLE
    assert result.uploaded_mb == result.downloaded_mb == 0.0
    assert result.elapsed_s == pytest.approx(REMOTE_PARAMETER_DEFAULTS["unreachable_detect_s"])
    assert result.prediction_mask is None


def test_packet_loss_above_ceiling_charges_the_lost_upload() -> None:
    result = transport().execute(request(payload_mb=1.0), RGB, LOSSY, timeout_s=60.0)
    assert result.status is RemoteStatus.UPLOAD_FAILED
    assert result.uploaded_mb == 1.0  # transmitted, then lost -- still charged
    assert result.downloaded_mb == 0.0
    assert result.elapsed_s == pytest.approx(0.02 + 1.0 * 8.0 / 50.0 + 0.040)


def test_timeout_fires_exactly_and_charges_partial_upload() -> None:
    # 2 MB at 2 Mbps -> 8 s upload alone; timeout at 3 s lands mid-upload.
    result = transport().execute(request(payload_mb=2.0), RGB, DEGRADED, timeout_s=3.0)
    assert result.status is RemoteStatus.TIMEOUT
    assert result.elapsed_s == 3.0
    expected_fraction = (3.0 - 0.02) / 8.0
    assert result.uploaded_mb == pytest.approx(2.0 * expected_fraction)
    assert result.downloaded_mb == 0.0
    assert result.prediction_mask is None


def test_backend_error_becomes_remote_error_after_upload() -> None:
    result = transport(_RaisingBackend()).execute(request(), RGB, GOOD, timeout_s=60.0)
    assert result.status is RemoteStatus.REMOTE_ERROR
    assert "server exploded" in result.failure_reason
    assert result.uploaded_mb == 2.0
    assert result.downloaded_mb == 0.0


def test_transport_is_deterministic() -> None:
    a = transport().execute(request(), RGB, DEGRADED, timeout_s=3.0)
    b = transport().execute(request(), RGB, DEGRADED, timeout_s=3.0)
    assert a.status == b.status and a.elapsed_s == b.elapsed_s
    assert a.uploaded_mb == b.uploaded_mb and a.stages_s == b.stages_s


# --- the deployment boundary -----------------------------------------------------------------


def test_request_is_wire_representable_with_stable_identity() -> None:
    wire = request().to_wire()
    json.dumps(wire)  # process-independent: JSON scalars/lists only
    assert wire["protocol_version"] == PROTOCOL_VERSION == "1.0"
    assert wire["request_id"] == request().to_wire()["request_id"]  # stable across builds
    assert all(not isinstance(v, np.generic | np.ndarray) for v in wire.values())


def test_local_and_remote_executors_share_the_result_shape(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_fast")
    local = runner.executor_for("FAST").run(RGB)
    remote = runner.executor_for("REMOTE_STRONG").run_with_context(
        RGB,
        ExecutionContext(
            scenario_id="S",
            observation_id=0,
            capture_time_s=0.0,
            deadline_s=60.0,
            network=GOOD,
        ),
    )
    # Same result contract; remote adds only the optional diagnostics block (the same
    # additive pattern the torch executors already use).
    assert set(remote.to_summary()) - set(local.to_summary()) == {"diagnostics"}
    assert set(local.to_summary()) <= set(remote.to_summary())
    assert local.communication_energy_j == 0.0  # local executors never charge radio


# --- communication and energy accounting ------------------------------------------------------


def remote_executor(make_scenario, **kwargs) -> SimulatedRemoteExecutor:
    scenario = load_scenario(make_scenario(**kwargs))
    runner = MissionRunner(scenario, "always_fast")
    return runner.executor_for("REMOTE_STRONG")


def ctx(network: NetworkRegime) -> ExecutionContext:
    return ExecutionContext(
        scenario_id="S", observation_id=3, capture_time_s=3.0, deadline_s=60.0, network=network
    )


def test_successful_call_charges_both_directions_and_radio_energy(make_scenario) -> None:
    result = remote_executor(make_scenario).run_with_context(RGB, ctx(GOOD))
    d = REMOTE_PARAMETER_DEFAULTS
    assert result.communication_mb == pytest.approx(2.0 + d["download_mb_per_call"])
    expected_energy = (
        2.0 * d["uplink_energy_j_per_mb"]
        + d["download_mb_per_call"] * d["downlink_energy_j_per_mb"]
        + d["radio_activation_j"]
    )
    assert result.communication_energy_j == pytest.approx(expected_energy)
    assert result.energy_j == pytest.approx(d["onboard_codec_energy_j"])


def test_unreachable_attempt_still_pays_radio_activation(make_scenario) -> None:
    result = remote_executor(make_scenario).run_with_context(RGB, ctx(DISCONNECTED))
    assert result.communication_mb == 0.0
    assert result.communication_energy_j == pytest.approx(
        REMOTE_PARAMETER_DEFAULTS["radio_activation_j"]
    )


def test_mission_charges_communication_against_the_budget(make_scenario) -> None:
    # A tiny budget: the always-remote baseline must violate the communication constraint.
    scenario = load_scenario(make_scenario(**{"mission_contract.communication_budget_mb": 5.0}))
    result = run_mission(scenario, "always_remote_strong")
    assert result.cumulative_communication_mb > 5.0
    assert result.constraints["communication_constraint_success"] is False
    assert result.mission_success is False
    assert result.communication_energy_j > 0.0


def test_local_missions_transmit_nothing(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_fast")
    assert result.cumulative_communication_mb == 0.0
    assert result.communication_energy_j == 0.0
    assert result.network_behaviour["remote_attempts"] == 0


# --- dynamic network model --------------------------------------------------------------------


def test_regime_boundaries_are_half_open_and_clamped() -> None:
    model = V2NetworkModel((GOOD, DEGRADED, DISCONNECTED, RECOVERED))
    assert model.state_at(0.0).regime_id == "5g_good"
    assert model.state_at(9.999).regime_id == "5g_good"
    assert model.state_at(10.0).regime_id == "lte_degraded"  # boundary joins the new regime
    assert model.state_at(23.999).regime_id == "disconnected"
    assert model.state_at(1_000.0).regime_id == "recovered"  # clamped past the end


def test_malformed_traces_fail_loudly() -> None:
    with pytest.raises(SchemaValidationError, match=r"start at 0\.0"):
        V2NetworkModel((DEGRADED,))
    with pytest.raises(SchemaValidationError, match="strictly increasing"):
        V2NetworkModel((GOOD, NetworkRegime("dup", 0.0, 1.0, 1.0, 1.0, 0.0)))


def test_policy_sees_only_the_current_sample(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    result = run_mission(scenario, "always_fast", record_runtime_snapshots=True)
    model = V2NetworkModel(scenario.simulation.network_trace)
    for log in result.observations:
        visible = log.runtime["at_capture"]["network"]
        # Exactly the frozen V1 observation fields: no regime name, no future, no trace.
        assert set(visible) == {"bandwidth_mbps", "rtt_ms", "packet_loss_frac"}
        expected = model.observation_at(log.capture_time_s)
        assert visible["bandwidth_mbps"] == expected.bandwidth_mbps


def test_identical_traces_produce_identical_missions(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    a = run_mission(scenario, "always_remote_strong")
    b = run_mission(scenario, "always_remote_strong")
    assert a.to_dict(include_observations=False) == b.to_dict(include_observations=False)


def test_constant_network_scenarios_are_untouched(make_scenario, tmp_path: Path) -> None:
    world = write_world_png(tmp_path / "w2.png")
    payload = scenario_payload(world)  # no trace, no remote config -- the frozen V2 shape
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.simulation.network_trace is None
    result = run_mission(scenario, "always_fast", record_runtime_snapshots=True)
    assert result.constraints["privacy_constraint_success"] is True  # fifth key, trivially met
    model = constant_network_model(*scenario.simulation.network)
    assert model.state_at(999.0).regime_id == "constant"
    first = result.observations[0].runtime["at_capture"]["network"]
    assert first["bandwidth_mbps"] == scenario.simulation.network[0]


# --- failure, timeout, and fallback semantics --------------------------------------------------


def test_failed_remote_without_fallback_is_not_an_empty_prediction(make_scenario) -> None:
    # Build a remote config without a fallback.
    scenario = load_scenario(make_scenario())
    spec = next(s for s in scenario.executor_configs if s.kind == "simulated_remote")
    no_fallback = ExecutorConfigSpec(
        config_id=spec.config_id,
        model_strategy_id=spec.model_strategy_id,
        kind=spec.kind,
        mission_latency_s=spec.mission_latency_s,
        energy_j_per_call=spec.energy_j_per_call,
        communication_mb_per_call=spec.communication_mb_per_call,
        quality_tier=spec.quality_tier,
        parameters={k: v for k, v in spec.parameters.items() if k != "fallback_config_id"},
    )
    runner = MissionRunner(scenario, "always_fast")
    executor = build_remote_executor(no_fallback, {"FAST": runner.executor_for("FAST")})
    result = executor.run_with_context(RGB, ctx(DISCONNECTED))
    assert result.success is False  # a failure, not a clean "nothing detected"
    assert not result.prediction_mask.any()
    assert result.diagnostics["remote_status"] == "unreachable"
    assert result.diagnostics["fallback"] is None


def test_fallback_runs_on_the_same_capture_and_charges_both_attempts(make_scenario) -> None:
    scenario = load_scenario(make_scenario())
    runner = MissionRunner(scenario, "always_fast")
    executor = runner.executor_for("REMOTE_STRONG")
    fast = runner.executor_for("FAST")

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:20, 10:30] = (230, 70, 20)  # marker-coloured patch the fast model detects
    result = executor.run_with_context(rgb, ctx(DISCONNECTED))

    assert result.success is True
    assert result.diagnostics["remote_status"] == "unreachable"
    assert result.diagnostics["fallback"]["fallback_config_id"] == "FAST"
    expected = fast.run(rgb)
    assert np.array_equal(result.prediction_mask, expected.prediction_mask)  # same frame
    assert result.mission_latency_s == pytest.approx(
        REMOTE_PARAMETER_DEFAULTS["unreachable_detect_s"] + expected.mission_latency_s
    )
    assert result.energy_j == pytest.approx(
        REMOTE_PARAMETER_DEFAULTS["onboard_codec_energy_j"] + expected.energy_j
    )


def test_runner_counts_remote_behaviour(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_remote_strong")
    behaviour = result.network_behaviour
    assert behaviour["remote_attempts"] > 0
    assert behaviour["remote_attempts"] == (
        behaviour["remote_successes"] + behaviour["remote_failures"]
    )
    assert behaviour["remote_unreachable"] > 0  # the disconnected regime was crossed
    assert behaviour["remote_fallbacks"] == behaviour["remote_failures"]  # fallback configured
    assert result.failed_inference_count == 0  # every failure fell back successfully


# --- privacy: the fifth hard constraint --------------------------------------------------------


def test_local_only_blocks_remote_and_fails_the_privacy_constraint(make_scenario) -> None:
    scenario = load_scenario(make_scenario(**{"mission_contract.privacy_level": "local_only"}))
    result = run_mission(scenario, "always_remote_strong")
    assert result.privacy_violation_count > 0
    assert result.constraints["privacy_constraint_success"] is False
    assert result.mission_success is False
    # The forbidden transfer never happened: the validator substituted a local config.
    assert set(result.config_selection_history) <= {"FAST", "STRONG"}
    assert result.cumulative_communication_mb == 0.0


def test_features_only_forbids_raw_rgb_upload(make_scenario) -> None:
    scenario = load_scenario(make_scenario(**{"mission_contract.privacy_level": "features_only"}))
    result = run_mission(scenario, "always_remote_strong")
    # The simulated remote transmits raw input, which features_only forbids.
    assert result.privacy_violation_count > 0
    assert result.constraints["privacy_constraint_success"] is False


def test_remote_allowed_permits_remote_without_violations(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "always_remote_strong")
    assert result.privacy_violation_count == 0
    assert result.constraints["privacy_constraint_success"] is True


def test_mission_success_is_the_five_constraint_conjunction(make_scenario) -> None:
    result = run_mission(load_scenario(make_scenario()), "rule_based")
    assert set(result.constraints) == {
        "quality_success",
        "deadline_success",
        "battery_constraint_success",
        "communication_constraint_success",
        "privacy_constraint_success",
    }
    assert result.mission_success == all(result.constraints.values())


def test_scenario_level_fallback_may_never_be_remote(make_scenario) -> None:
    """The safe fallback is the last line of defence; it may not depend on the network."""
    with pytest.raises(SchemaValidationError, match="local configuration"):
        load_scenario(make_scenario(**{"simulation.fallback_config_id": "REMOTE_STRONG"}))


# --- network-aware policy behaviour ------------------------------------------------------------


def test_rule_based_dormant_network_rules_are_now_exercised(make_scenario) -> None:
    """The V1 reachability/latency rules fire: remote on good links, local on bad ones."""
    result = run_mission(load_scenario(make_scenario()), "rule_based")
    history = result.config_selection_history
    assert "REMOTE_STRONG" in history  # used while the link was good
    assert "FAST" in history  # abandoned remote when the link degraded/died
    assert result.network_behaviour["remote_failures"] == 0  # it never tried a dead link
    assert result.mission_success is True


# --- replay integration -----------------------------------------------------------------------


def test_replay_bundle_exposes_remote_and_privacy_diagnostics(make_scenario, tmp_path) -> None:
    from aerointentbench.v2.replay_export import export_replay_bundle

    bundle = export_replay_bundle(make_scenario(), "always_remote_strong", tmp_path / "bundle")
    events = json.loads((bundle / "events.json").read_text())
    remote_events = [
        e
        for e in events["events"]
        if (e["execution"].get("diagnostics") or {}).get("remote_status")
    ]
    assert remote_events, "remote executions must be visible in the replay"
    sample = remote_events[0]["execution"]["diagnostics"]
    assert {"remote_status", "network", "latency_breakdown_s", "uploaded_mb"} <= set(sample)
    for event in events["events"]:
        assert event["constraint_status"]["privacy"]["status"] in ("SAFE", "VIOLATED")
    assert events["final"]["constraint_status"]["privacy"] in ("SAFE", "VIOLATED")
    html = (bundle / "index.html").read_text()
    assert "remote status" in html  # the viewer can show the remote path
