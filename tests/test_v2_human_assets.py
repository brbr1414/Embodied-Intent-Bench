"""V2.2 image-asset targets: manifest, loading, compositing, GT, and observability.

Everything here runs on tiny procedurally generated RGBA fixtures and fake model
backends — no internet, no weights, no GeoTIFF, no licensed assets. The procedural
silhouette used below is a test fixture, never a realistic person and never evidence of
model performance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="V2 tests need the aerointentbench[v2] extras")
pytest.importorskip("PIL", reason="V2 tests need the aerointentbench[v2] extras")

from aerointentbench.schemas.loading import (  # noqa: E402
    SchemaValidationError,
    SchemaVersionError,
)
from aerointentbench.v2.assets import (  # noqa: E402
    clear_asset_cache,
    load_asset,
    load_manifest,
)
from aerointentbench.v2.camera import CameraRenderer  # noqa: E402
from aerointentbench.v2.objects import ObjectLayer  # noqa: E402
from aerointentbench.v2.scenario import ObjectSpec, load_scenario  # noqa: E402
from aerointentbench.v2.world import ArrayWorld  # noqa: E402
from tests.test_v2_visual_loop import (  # noqa: E402
    BG,
    scenario_payload,
    write_world_png,
)

# --- fixtures --------------------------------------------------------------------------------


def silhouette_rgba(width_px: int = 40, height_px: int = 80) -> np.ndarray:
    """A human-like test silhouette: head circle + body ellipse, transparent elsewhere.

    A *procedural test fixture* -- explicitly not a realistic person target.
    """
    rgba = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    cx = width_px / 2.0
    head = ((xx - cx) ** 2 + (yy - height_px * 0.15) ** 2) <= (height_px * 0.10) ** 2
    body = (
        ((xx - cx) / (width_px * 0.28)) ** 2 + ((yy - height_px * 0.55) / (height_px * 0.38)) ** 2
    ) <= 1.0
    inside = head | body
    rgba[inside] = (40, 60, 150, 255)  # clothed-figure blue, fully opaque
    return rgba


def write_asset(path: Path, rgba: np.ndarray | None = None) -> str:
    """Write an RGBA PNG and return its sha256."""
    from PIL import Image

    rgba = silhouette_rgba() if rgba is None else rgba
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload(asset_path: Path, checksum: str, **overrides) -> dict:
    record = {
        "asset_id": "test_silhouette",
        "asset_path": asset_path.name,
        "mask_source": "alpha",
        "category": "person",
        "view_type": "procedural",
        "source": {
            "provider": "generated test fixture",
            "license": "CC0-1.0",
            "redistribution_allowed": True,
        },
        "physical": {"nominal_width_m": 0.6, "nominal_height_m": 1.75},
        "checksum_sha256": checksum,
    }
    record.update(overrides)
    return {"asset_schema_version": "1.0", "assets": [record]}


@pytest.fixture
def asset_manifest(tmp_path: Path) -> Path:
    asset_path = tmp_path / "assets" / "sil.png"
    checksum = write_asset(asset_path)
    manifest = tmp_path / "assets" / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload(asset_path, checksum)), encoding="utf-8")
    clear_asset_cache()
    return manifest


def make_layer(manifest_path: Path, obj: ObjectSpec) -> ObjectLayer:
    manifest = load_manifest(manifest_path)
    assets = {obj.asset_id: load_asset(manifest.get(obj.asset_id))}
    return ObjectLayer((obj,), assets=assets)


def image_object(**overrides) -> ObjectSpec:
    base = dict(
        object_id="TGT_IMG",
        class_id="person",
        is_target=True,
        position_m=(50.0, 40.0),
        width_m=4.0,
        height_m=8.0,
        rotation_deg=0.0,
        z_order=10,
        render_mode="image_asset",
        asset_id="test_silhouette",
    )
    base.update(overrides)
    return ObjectSpec(**base)


def render_once(manifest_path: Path, obj: ObjectSpec, *, position=(50.0, 40.0)):
    from aerointentbench.v2.scenario import CameraSpec

    world = ArrayWorld(np.full((300, 400, 3), BG, dtype=np.uint8), meters_per_pixel=0.5)
    camera = CameraSpec(
        projection="orthographic",
        footprint_width_m=40.0,
        footprint_height_m=30.0,
        output_width_px=160,
        output_height_px=120,
    )
    renderer = CameraRenderer(
        scenario_id="T", world=world, objects=make_layer(manifest_path, obj), camera=camera
    )
    return renderer.render(position_m=position, capture_time_s=0.0, observation_id=0)


# --- manifest validation ---------------------------------------------------------------------


def test_a_valid_manifest_loads_and_decodes(asset_manifest: Path) -> None:
    manifest = load_manifest(asset_manifest)
    asset = load_asset(manifest.get("test_silhouette"))
    assert asset.mask.any() and asset.alpha.max() == 1.0
    assert asset.record.view_type == "procedural"
    assert asset.record.license == "CC0-1.0"


def test_unsupported_manifest_version_fails(tmp_path: Path, asset_manifest: Path) -> None:
    payload = json.loads(asset_manifest.read_text())
    payload["asset_schema_version"] = "9.9"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        load_manifest(bad)


def test_missing_asset_file_is_actionable(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(manifest_payload(tmp_path / "nope.png", "0" * 64)), encoding="utf-8"
    )
    with pytest.raises(SchemaValidationError, match="licensed asset"):
        load_manifest(manifest)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    asset_path = tmp_path / "sil.png"
    write_asset(asset_path)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(manifest_payload(asset_path, "ab" * 32)), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="checksum mismatch"):
        load_manifest(manifest)


def test_missing_license_metadata_is_rejected(tmp_path: Path) -> None:
    asset_path = tmp_path / "sil.png"
    checksum = write_asset(asset_path)
    payload = manifest_payload(asset_path, checksum)
    del payload["assets"][0]["source"]["license"]
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="license"):
        load_manifest(manifest)


def test_a_fabricated_view_type_is_rejected(tmp_path: Path) -> None:
    asset_path = tmp_path / "sil.png"
    checksum = write_asset(asset_path)
    payload = manifest_payload(asset_path, checksum, view_type="cinematic")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="view_type"):
        load_manifest(manifest)


def test_an_empty_mask_is_rejected(tmp_path: Path) -> None:
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)  # fully transparent
    asset_path = tmp_path / "empty.png"
    checksum = write_asset(asset_path, rgba)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(manifest_payload(asset_path, checksum)), encoding="utf-8")
    clear_asset_cache()
    loaded = load_manifest(manifest)
    with pytest.raises(SchemaValidationError, match="empty"):
        load_asset(loaded.get("test_silhouette"))


def test_separate_mask_file_dimension_mismatch_fails(tmp_path: Path) -> None:
    from PIL import Image

    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((40, 30, 3), 120, dtype=np.uint8)).save(rgb_path)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((20, 30), 255, dtype=np.uint8)).save(mask_path)
    checksum = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
    payload = manifest_payload(rgb_path, checksum, mask_source="file", mask_path="mask.png")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    clear_asset_cache()
    loaded = load_manifest(manifest)
    with pytest.raises(SchemaValidationError, match="dimensions must match"):
        load_asset(loaded.get("test_silhouette"))


# --- compositing and GT ----------------------------------------------------------------------


def test_asset_composites_with_gt_from_the_mask_not_the_rgb(asset_manifest: Path) -> None:
    obs = render_once(asset_manifest, image_object())
    target = obs.semantic_gt == 1
    assert target.any()
    # GT comes from the transformed asset mask: where GT is set, the RGB shows the
    # silhouette colour (blue-ish), not the background.
    assert (obs.rgb[target][:, 2].astype(int).mean()) > (obs.rgb[~target][:, 2].astype(int).mean())
    assert obs.instance_gt[target].min() == obs.instance_gt[target].max() == 1
    assert obs.visible_target_ids == ("TGT_IMG",)


def test_physical_size_drives_projected_pixels(asset_manifest: Path) -> None:
    # Footprint 40 m -> 160 px means 0.25 m/px: a 4x8 m object projects to 16x32 px.
    obs = render_once(asset_manifest, image_object(width_m=4.0, height_m=8.0))
    stats = obs.provenance["asset_projections"]["TGT_IMG"]
    assert stats["projected_width_px"] == 16.0
    assert stats["projected_height_px"] == 32.0
    assert 0 < stats["visible_px"] <= 16 * 32
    assert stats["visible_fraction"] == pytest.approx(1.0)


def test_zero_pixel_projection_is_recorded_never_enlarged(asset_manifest: Path) -> None:
    obs = render_once(asset_manifest, image_object(width_m=0.1, height_m=0.2))
    assert not (obs.semantic_gt == 1).any()
    stats = obs.provenance["asset_projections"]["TGT_IMG"]
    assert stats["projected_width_px"] == 0.0
    assert obs.visible_target_ids == ()


def test_rotation_keeps_rgb_and_gt_aligned(asset_manifest: Path) -> None:
    upright = render_once(asset_manifest, image_object(rotation_deg=0.0))
    rotated = render_once(asset_manifest, image_object(rotation_deg=90.0))
    up_mask, rot_mask = upright.semantic_gt == 1, rotated.semantic_gt == 1
    assert up_mask.any() and rot_mask.any()
    assert not np.array_equal(up_mask, rot_mask), "rotation must move the GT"

    # Alignment: under 90 deg, the mask's aspect flips (tall silhouette becomes wide).
    def extent(mask):
        ys, xs = np.where(mask)
        return (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

    up_h, up_w = extent(up_mask)
    rot_h, rot_w = extent(rot_mask)
    assert up_h > up_w and rot_w > rot_h
    # And the painted RGB follows the mask in both cases.
    for obs in (upright, rotated):
        mask = obs.semantic_gt == 1
        assert (obs.rgb[mask][:, 2].astype(int).mean()) > 100


def test_boundary_clipping_keeps_partial_assets(asset_manifest: Path) -> None:
    # Object centred at the footprint's left edge: half of it is outside the crop.
    obs = render_once(asset_manifest, image_object(position_m=(30.0, 40.0)))
    target = obs.semantic_gt == 1
    assert target.any()
    stats = obs.provenance["asset_projections"]["TGT_IMG"]
    assert stats["visible_fraction"] < 0.7  # clipped, and honestly reported
    cols = np.where(target.any(axis=0))[0]
    assert cols.min() == 0, "clipped asset must touch the crop edge"
    assert "TGT_IMG" in obs.visible_target_ids


def test_z_order_between_asset_and_marker(asset_manifest: Path) -> None:
    marker = ObjectSpec(
        object_id="DIS_MARKER",
        class_id="debris",
        is_target=False,
        position_m=(50.0, 40.0),
        width_m=6.0,
        height_m=3.0,
        rotation_deg=0.0,
        z_order=20,
        appearance={"shape": "rect", "body_rgb": (146, 95, 100), "stripe": False, "alpha": 1.0},
    )
    target = image_object(z_order=10)
    manifest = load_manifest(asset_manifest)
    layer = ObjectLayer(
        (target, marker), assets={"test_silhouette": load_asset(manifest.get("test_silhouette"))}
    )
    from aerointentbench.v2.scenario import CameraSpec

    world = ArrayWorld(np.full((300, 400, 3), BG, dtype=np.uint8), meters_per_pixel=0.5)
    renderer = CameraRenderer(
        scenario_id="T",
        world=world,
        objects=layer,
        camera=CameraSpec(
            projection="orthographic",
            footprint_width_m=40.0,
            footprint_height_m=30.0,
            output_width_px=160,
            output_height_px=120,
        ),
    )
    obs = renderer.render(position_m=(50.0, 40.0), capture_time_s=0.0, observation_id=0)
    overlap = (obs.instance_gt == layer.instance_id("DIS_MARKER")) & (obs.semantic_gt == 2)
    assert overlap.any(), "the higher z-order marker owns the overlap"
    assert (obs.semantic_gt == 1).any(), "the asset's unoccluded part remains target GT"


def test_rendering_is_deterministic(asset_manifest: Path) -> None:
    one = render_once(asset_manifest, image_object(rotation_deg=35.0, brightness_factor=1.2))
    two = render_once(asset_manifest, image_object(rotation_deg=35.0, brightness_factor=1.2))
    assert np.array_equal(one.rgb, two.rgb)
    assert np.array_equal(one.instance_gt, two.instance_gt)


def test_brightness_transform_touches_rgb_only(asset_manifest: Path) -> None:
    plain = render_once(asset_manifest, image_object())
    bright = render_once(asset_manifest, image_object(brightness_factor=1.8))
    mask = plain.semantic_gt == 1
    assert np.array_equal(mask, bright.semantic_gt == 1), "GT geometry must not change"
    assert bright.rgb[mask].astype(int).sum() > plain.rgb[mask].astype(int).sum()


def test_opacity_blends_rgb_without_changing_gt(asset_manifest: Path) -> None:
    solid = render_once(asset_manifest, image_object())
    faint = render_once(asset_manifest, image_object(opacity=0.4))
    mask = solid.semantic_gt == 1
    assert np.array_equal(mask, faint.semantic_gt == 1)
    # Lower opacity pulls the composited pixels toward the background colour.
    assert (
        np.abs(faint.rgb[mask].astype(int) - np.array(BG)).sum()
        < np.abs(solid.rgb[mask].astype(int) - np.array(BG)).sum()
    )


# --- scenario integration --------------------------------------------------------------------


def scenario_with_asset(tmp_path: Path, asset_manifest: Path, **object_overrides) -> Path:
    world = write_world_png(tmp_path / "world.png")
    payload = scenario_payload(world)
    payload["assets_manifest"] = str(asset_manifest)
    payload["evaluation_purpose"] = "integration_diagnostic"
    payload["objects"][0] = {
        "object_id": "TGT_1",
        "class_id": "person",
        "is_target": True,
        "render_mode": "image_asset",
        "asset_id": "test_silhouette",
        "position_m": [50.0, 40.0],
        "width_m": 6.0,
        "height_m": 12.0,
        "rotation_deg": 0.0,
        "z_order": 10,
        **object_overrides,
    }
    path = tmp_path / "asset_scenario.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_image_asset_scenario_loads_and_runs_with_fakes(
    tmp_path: Path, asset_manifest: Path
) -> None:
    from aerointentbench.v2.runner import run_mission

    scenario = load_scenario(scenario_with_asset(tmp_path, asset_manifest))
    assert scenario.evaluation_purpose == "integration_diagnostic"
    result = run_mission(scenario, "always_strong")  # V2.0 heuristic executors still work
    assert result.processed_observation_count > 0
    # The heuristic strong executor looks for orange; the blue silhouette is invisible
    # to it -- but the mission and GT accounting run end to end.
    assert result.quality["targets_encountered"] >= 1


def test_image_asset_without_manifest_fails(tmp_path: Path, asset_manifest: Path) -> None:
    path = scenario_with_asset(tmp_path, asset_manifest)
    payload = json.loads(path.read_text())
    del payload["assets_manifest"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="assets_manifest"):
        load_scenario(path)


def test_a_non_person_asset_cannot_be_a_target(tmp_path: Path, asset_manifest: Path) -> None:
    from aerointentbench.v2.runner import MissionRunner

    payload = json.loads(asset_manifest.read_text())
    payload["assets"][0]["category"] = "vehicle"
    manifest2 = asset_manifest.parent / "vehicle_manifest.json"
    manifest2.write_text(json.dumps(payload), encoding="utf-8")
    scenario = load_scenario(scenario_with_asset(tmp_path, manifest2))
    with pytest.raises(SchemaValidationError, match="person"):
        MissionRunner(scenario, "always_fast")


def test_unknown_asset_id_fails(tmp_path: Path, asset_manifest: Path) -> None:
    from aerointentbench.v2.runner import MissionRunner

    scenario = load_scenario(scenario_with_asset(tmp_path, asset_manifest, asset_id="ghost_asset"))
    with pytest.raises(SchemaValidationError, match="ghost_asset"):
        MissionRunner(scenario, "always_fast")


def test_executor_and_policy_receive_no_asset_metadata(
    tmp_path: Path, asset_manifest: Path
) -> None:
    import inspect

    from aerointentbench.v2 import executors as executors_module
    from aerointentbench.v2.runner import run_mission

    source = inspect.getsource(executors_module)
    for forbidden in ("asset_id", "LoadedAsset", "view_type", "license"):
        assert forbidden not in source, f"executors must not reference {forbidden}"

    scenario = load_scenario(scenario_with_asset(tmp_path, asset_manifest))
    result = run_mission(scenario, "always_fast")
    for log in result.observations:
        blob = json.dumps(log.to_dict()["execution"]) + json.dumps(
            {k: v for k, v in log.to_dict().items() if k != "score"}
        )
        for token in ("asset_id", "view_type", "license", "silhouette"):
            assert token not in blob


# --- observability experiment (fake backends) ------------------------------------------------


def observability_config(tmp_path: Path, asset_manifest: Path) -> Path:
    world = write_world_png(tmp_path / "world.png")
    payload = {
        "observability_schema_version": "1.0",
        "experiment_id": "OBS_TEST",
        "evaluation_purpose": "controlled_observability",
        "world": {
            "image_path": str(world),
            "source_id": "test_world",
            "meters_per_pixel": 0.5,
        },
        "camera": {
            "projection": "orthographic",
            "footprint_width_m": 40.0,
            "footprint_height_m": 30.0,
            "output_width_px": 160,
            "output_height_px": 120,
        },
        "assets_manifest": str(asset_manifest),
        "matching_iou_threshold": 0.3,
        "executor_configs": scenario_payload(world)["executor_configs"],
        "backgrounds": [
            {"background_id": "plain", "position_m": [50.0, 40.0], "description": "flat gray"},
        ],
        "conditions": [
            {
                "condition_id": "large_r0",
                "asset_id": "test_silhouette",
                "background_id": "plain",
                "width_m": 6.0,
                "height_m": 12.0,
                "rotation_deg": 0.0,
            },
            {
                "condition_id": "small_r90",
                "asset_id": "test_silhouette",
                "background_id": "plain",
                "width_m": 1.0,
                "height_m": 2.0,
                "rotation_deg": 90.0,
            },
        ],
    }
    path = tmp_path / "obs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_observability_report_generation(tmp_path: Path, asset_manifest: Path) -> None:
    from aerointentbench.v2.observability import load_config, run_experiment

    config = load_config(observability_config(tmp_path, asset_manifest))
    report = run_experiment(config, tmp_path / "out", panels=1)
    assert report["evaluation_purpose"] == "controlled_observability"
    assert report["honesty"]["view_types_present"] == ["procedural"]
    rows = report["rows"]
    assert len(rows) == 4  # 2 conditions x 2 heuristic executors
    for row in rows:
        for key in (
            "condition_id",
            "asset_id",
            "view_type",
            "background_id",
            "projected_width_px",
            "target_visible_px",
            "model_strategy",
            "prediction_positive_px",
            "target_pixel_iou",
            "target_pixel_recall",
            "false_positive_px",
            "detected_by_evidence_rule",
        ):
            assert key in row, key
    # The blue silhouette is invisible to the orange-gated heuristics: honest zeros.
    assert all(row["target_intersection_px"] == 0 for row in rows)
    assert (tmp_path / "out" / "OBS_TEST_observability.json").is_file()
    assert list((tmp_path / "out").glob("*.png")), "one debug panel was requested"


def test_observability_cli(tmp_path: Path, asset_manifest: Path, capsys) -> None:
    from aerointentbench.v2.cli import main

    config = observability_config(tmp_path, asset_manifest)
    assert (
        main(["run-observability", "--config", str(config), "--output", str(tmp_path / "o")]) == 0
    )
    out = capsys.readouterr().out
    assert "NOT real aerial-human perception" in out


def test_asset_cli_validate_and_inspect(tmp_path: Path, asset_manifest: Path, capsys) -> None:
    from aerointentbench.v2.cli import main

    assert main(["validate-assets", "--manifest", str(asset_manifest)]) == 0
    assert "procedural" in capsys.readouterr().out
    assert (
        main(["inspect-assets", "--manifest", str(asset_manifest), "--output", str(tmp_path / "i")])
        == 0
    )
    assert list((tmp_path / "i").glob("asset_*.png"))


def test_the_committed_observability_configs_parse() -> None:
    """Every shipped observability config must at least parse without the local assets.

    The manifests they reference point at owner-supplied processed PNGs that are
    local-only (redistribution pending owner confirmation, gitignored); resolving those
    files is the opt-in real_assets test below.
    """
    from aerointentbench.v2.observability import load_config

    repo = Path(__file__).resolve().parents[1]
    for name in (
        "observability_img1.json",
        "observability_generated_stageA.json",
        "observability_generated_stageB.json",
    ):
        config = load_config(repo / "data" / "v2_scenarios" / name)
        assert config.evaluation_purpose == "controlled_observability"


@pytest.mark.real_assets
def test_the_committed_manifest_resolves_the_local_generated_assets() -> None:
    """With the owner-supplied assets present, the shipped manifest and every condition
    asset id in the shipped configs must resolve, and every asset must be synthetic and
    honestly view-typed."""
    from aerointentbench.v2.observability import load_config

    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / "data" / "v2_assets" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("owner-supplied manifest not present")
    manifest = load_manifest(manifest_path)
    assert all(record.synthetic for record in manifest.records.values())
    assert {r.view_type for r in manifest.records.values()} <= {"conventional", "aerial"}
    for name in (
        "observability_img1.json",
        "observability_generated_stageA.json",
        "observability_generated_stageB.json",
    ):
        config = load_config(repo / "data" / "v2_scenarios" / name)
        for condition in config.conditions:
            assert condition.asset_id in manifest, (name, condition.asset_id)


# --- deterministic asset preparation (asset_prep) --------------------------------------------


def test_halo_removal_keeps_the_aa_band_and_zeroes_the_halo() -> None:
    from aerointentbench.v2.asset_prep import clean_rgba

    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[20:40, 20:40] = (200, 50, 50, 255)  # solid subject
    rgba[19, 20:40] = (200, 50, 50, 120)  # genuine AA edge (within 3px band)
    rgba[2:58, 2:58, 3] = np.maximum(rgba[2:58, 2:58, 3], 20)  # broad low-alpha halo
    result = clean_rgba(rgba)
    alpha = result.rgba[..., 3]
    assert (alpha == 120).any(), "the anti-aliased boundary must survive"
    # Everything beyond the keep band is zero: total faint pixels shrink to the band.
    assert int((alpha == 20).sum()) < 400, "the broad halo must be gone"
    assert result.solid_px == 20 * 20


def test_cleaning_is_deterministic_and_crops_with_padding() -> None:
    from aerointentbench.v2.asset_prep import clean_rgba

    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[40:60, 45:55] = (10, 200, 10, 255)
    one, two = clean_rgba(rgba), clean_rgba(rgba)
    assert np.array_equal(one.rgba, two.rgba)
    assert one.rgba.shape == (20 + 16, 10 + 16, 4)  # bbox + 8px pad each side
    assert one.processing.startswith("halo-removal")


def test_an_asset_with_no_solid_subject_is_rejected() -> None:
    from aerointentbench.v2.asset_prep import clean_rgba

    rgba = np.zeros((30, 30, 4), dtype=np.uint8)
    rgba[..., 3] = 40  # faint everywhere, never solid
    with pytest.raises(SchemaValidationError, match="no solid subject"):
        clean_rgba(rgba)


def test_prepare_asset_writes_and_checksums(tmp_path: Path) -> None:
    from aerointentbench.v2.asset_prep import inspect_rgba, prepare_asset

    src = tmp_path / "src.png"
    write_asset(src)  # the procedural silhouette fixture
    info = inspect_rgba(src)
    assert info["has_alpha"] and info["solid_area_fraction"] > 0
    facts = prepare_asset(src, tmp_path / "out" / "clean.png")
    assert (tmp_path / "out" / "clean.png").is_file()
    assert len(facts["checksum_sha256"]) == 64
    assert facts["solid_px"] > 0


def test_manifest_pose_and_synthetic_fields_roundtrip(tmp_path: Path) -> None:
    asset_path = tmp_path / "sil.png"
    checksum = write_asset(asset_path)
    payload = manifest_payload(asset_path, checksum)
    payload["assets"][0]["pose"] = "standing"
    payload["assets"][0]["synthetic"] = True
    payload["assets"][0]["processing"] = "halo-removal(...); crop"
    manifest_file = tmp_path / "m.json"
    manifest_file.write_text(json.dumps(payload), encoding="utf-8")
    clear_asset_cache()
    record = load_manifest(manifest_file).get("test_silhouette")
    assert record.pose == "standing" and record.synthetic is True
    assert record.provenance()["synthetic"] is True
    assert record.processing.startswith("halo-removal")
