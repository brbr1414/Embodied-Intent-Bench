"""V3 P4: UAVid adapter (official layout, windowed) and tiled-model plumbing, CI-safe.

No real dataset, no torch: the adapter is exercised on tiny synthetic label PNGs in
the official UAVid directory layout, and only the tiled model's pure-Python pieces
(tile grid, lazy import boundary) are touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="pilot adapter tests need the [v2] extras")
pytest.importorskip("PIL", reason="pilot adapter tests need Pillow")

from experiments.real_segmentation_pilot.torchvision_models import TiledTorchvisionModel
from experiments.real_segmentation_pilot.uavid import HUMAN_RGB, UAVidAdapter

# --- fixture: a tiny official-layout UAVid tree ----------------------------------------------


def _write_frame(
    root: Path,
    split: str,
    sequence: str,
    name: str,
    human_blocks: list[tuple[int, int, int, int]],
    *,
    size=(40, 60),
) -> None:
    """One Images/Labels pair; human_blocks are (top, left, h, w) rectangles."""
    import numpy as np
    from PIL import Image

    height, width = size
    base = root / split / split / sequence
    (base / "Images").mkdir(parents=True, exist_ok=True)
    (base / "Labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (90, 110, 90)).save(base / "Images" / f"{name}.png")
    label = np.zeros((height, width, 3), dtype=np.uint8)
    label[:, :] = (128, 64, 128)  # road background
    for top, left, h, w in human_blocks:
        label[top : top + h, left : left + w] = HUMAN_RGB
    Image.fromarray(label).save(base / "Labels" / f"{name}.png")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "uavid"
    _write_frame(root, "train", "seq1", "000100", [(5, 10, 2, 3), (20, 30, 3, 3)])
    _write_frame(root, "train", "seq1", "000200", [(8, 40, 2, 2)])
    _write_frame(root, "valid", "seq2", "000100", [])  # no humans -> excluded
    return root


FULL = (40, 60)  # window == frame: the whole tiny frame is evaluated


# --- adapter ---------------------------------------------------------------------------------


def test_adapter_derives_instances_from_the_semantic_colour(dataset_root: Path) -> None:
    adapter = UAVidAdapter(dataset_root, window=FULL)
    frames = adapter.frames()
    truth = adapter.ground_truth()

    assert [f.source_metadata["image_name"] for f in frames] == [
        "seq1/000100",
        "seq1/000200",
    ]
    assert [f.frame_id for f in frames] == [0, 1]
    assert adapter.image_width == 60 and adapter.image_height == 40
    # Frames carry the evaluated crop itself, at the window size.
    assert frames[0].image.shape == (40, 60, 3)

    by_frame: dict[int, list] = {}
    for gt in truth:
        by_frame.setdefault(gt.frame_id, []).append(gt)
    assert len(by_frame[0]) == 2 and len(by_frame[1]) == 1
    first = next(gt for gt in by_frame[0] if gt.track_id == "seq1/000100#000")
    assert (5, 10) in first.mask.pixels and (6, 12) in first.mask.pixels
    assert first.mask.area == 6
    assert all(gt.category == "person" for gt in truth)


def test_adapter_track_ids_are_per_frame_identities(dataset_root: Path) -> None:
    adapter = UAVidAdapter(dataset_root, window=FULL)
    ids = [gt.track_id for gt in adapter.ground_truth()]
    assert len(ids) == len(set(ids))
    provenance = adapter.provenance()
    assert "temporal_identity" in provenance
    assert "instance_derivation" in provenance
    assert provenance["human_instances"] == 3


def test_adapter_caps_frames_toward_the_target_density(dataset_root: Path) -> None:
    adapter = UAVidAdapter(dataset_root, window=FULL, max_frames=1, target_instances_per_frame=2)
    frames = adapter.frames()
    # seq1/000100 has 2 derived instances -- exactly the target -- so it wins the cap.
    assert [f.source_metadata["image_name"] for f in frames] == ["seq1/000100"]


def test_window_is_centred_on_annotations_and_masks_are_window_relative(
    tmp_path: Path,
) -> None:
    root = tmp_path / "uavid"
    # One 4x4 human block far from the frame centre.
    _write_frame(root, "train", "seq1", "000100", [(30, 46, 4, 4)])
    adapter = UAVidAdapter(root, splits=("train",), window=(10, 12))
    provenance = adapter.provenance()["windows"][0]
    # Median annotated pixel (31, 47); window (10x12) centred there: (31-5, 47-6).
    assert provenance["window_top"] == 26 and provenance["window_left"] == 41
    (gt,) = adapter.ground_truth()
    assert gt.mask.height == 10 and gt.mask.width == 12
    # Full-frame (30, 46) maps to window-relative (4, 5).
    assert (4, 5) in gt.mask.pixels and gt.mask.area == 16
    frame = adapter.frames()[0]
    assert frame.image.shape == (10, 12, 3)
    assert frame.source_metadata["window_top"] == 26


def test_instances_outside_the_window_are_counted_as_clipped(tmp_path: Path) -> None:
    root = tmp_path / "uavid"
    # A cluster the window will centre on, plus one distant human it cannot cover.
    _write_frame(
        root,
        "train",
        "seq1",
        "000100",
        [(10, 10, 3, 3), (12, 16, 3, 3), (35, 55, 3, 3)],
    )
    adapter = UAVidAdapter(root, splits=("train",), window=(12, 16))
    provenance = adapter.provenance()
    assert provenance["human_instances"] == 2
    assert provenance["clipped_instances_total"] == 1


def test_frames_smaller_than_the_window_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "uavid"
    _write_frame(root, "train", "seq1", "000100", [(5, 10, 2, 3)], size=(20, 30))
    with pytest.raises(ValueError, match="no frames with Humans"):
        UAVidAdapter(root, splits=("train",), window=(40, 60))


def test_adapter_fails_loudly_on_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UAVidAdapter(tmp_path / "nowhere")


def test_adapter_fails_loudly_when_nothing_has_humans(tmp_path: Path) -> None:
    root = tmp_path / "uavid"
    _write_frame(root, "train", "seq1", "000100", [])
    with pytest.raises(ValueError, match="no frames with Humans"):
        UAVidAdapter(root, splits=("train",), window=FULL)


# --- tiled model plumbing (no torch) ---------------------------------------------------------


def test_tile_grid_covers_the_frame_with_inward_edge_tiles() -> None:
    model = TiledTorchvisionModel(
        config_id="CFG",
        model_id="lraspp_mobilenet_v3_large",
        tile_px=512,
        probability_threshold=0.5,
    )
    origins = model._tile_origins(720, 1280)
    assert (0, 0) in origins
    assert (720 - 512, 1280 - 512) in origins  # inward-shifted final tiles
    rows = {top for top, _left in origins}
    cols = {left for _top, left in origins}
    assert max(rows) + 512 == 720 and max(cols) + 512 == 1280
    assert min(rows) == 0 and min(cols) == 0
    assert len(origins) == len(rows) * len(cols)


def test_small_frame_gets_a_single_origin() -> None:
    model = TiledTorchvisionModel(
        config_id="CFG",
        model_id="deeplabv3_resnet50",
        tile_px=512,
        probability_threshold=0.5,
    )
    assert model._tile_origins(200, 300) == [(0, 0)]


def test_constructing_the_model_needs_no_torch() -> None:
    # The lazy boundary: constructing the dataclass must not import torch.
    TiledTorchvisionModel(
        config_id="CFG",
        model_id="deeplabv3_resnet50",
        tile_px=512,
        probability_threshold=0.5,
    )
