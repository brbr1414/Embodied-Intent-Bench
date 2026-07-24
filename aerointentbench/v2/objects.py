"""The synthetic object layer: targets and distractors composited over the aerial world.

The source raster is never modified. Objects live in world coordinates and are painted
into each camera crop at render time, together with the semantic and instance ground
truth that scoring needs. **These are synthetic rescue-target markers, not realistic
humans** -- a recognisable high-visibility pattern the lightweight V2 executors can find
from RGB alone, with distractors wearing similar-but-imperfect appearances.

Rendering supports rotation, partial visibility at crop boundaries, deterministic
z-order (higher ``z_order`` paints later; ties break on ``object_id``), alpha blending
into the RGB, and exact per-pixel masks. Semantic labels: 0 background, 1 target,
2 distractor. Instance labels: 0 background, else 1-based index into the scenario's
object list (stable across the mission).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from aerointentbench.v2.scenario import ObjectSpec

__all__ = [
    "SEMANTIC_BACKGROUND",
    "SEMANTIC_DISTRACTOR",
    "SEMANTIC_TARGET",
    "ObjectLayer",
    "RenderedObjects",
]

SEMANTIC_BACKGROUND: Final = 0
SEMANTIC_TARGET: Final = 1
SEMANTIC_DISTRACTOR: Final = 2


@dataclass(frozen=True, slots=True)
class RenderedObjects:
    """What the object layer contributed to one crop."""

    semantic: np.ndarray  # uint8 (H, W): 0 bg / 1 target / 2 distractor
    instance: np.ndarray  # int32 (H, W): 0 bg, else 1-based object ordinal
    visible_target_ids: tuple[str, ...]
    visible_distractor_ids: tuple[str, ...]


class ObjectLayer:
    """Holds the scenario's objects and paints the ones a footprint can see."""

    def __init__(self, objects: tuple[ObjectSpec, ...]) -> None:
        # Paint order: z_order then object_id -- fully deterministic under overlap.
        self._objects = tuple(sorted(objects, key=lambda o: (o.z_order, o.object_id)))
        #: 1-based instance ids follow the *scenario declaration* order, not paint order,
        #: so an id is stable regardless of z-order edits.
        self._instance_ids = {obj.object_id: i + 1 for i, obj in enumerate(objects)}

    @property
    def objects(self) -> tuple[ObjectSpec, ...]:
        return self._objects

    def instance_id(self, object_id: str) -> int:
        return self._instance_ids[object_id]

    @property
    def instance_object_ids(self) -> dict[str, str]:
        """The full ``str(instance_id) -> object_id`` map, for observation provenance."""
        return {str(i): object_id for object_id, i in self._instance_ids.items()}

    # -- visibility ---------------------------------------------------------------------

    def intersecting(
        self, footprint_m: tuple[float, float, float, float]
    ) -> tuple[ObjectSpec, ...]:
        """Objects whose bounding circle touches the footprint rectangle."""
        fx0, fy0, fx1, fy1 = footprint_m
        hits = []
        for obj in self._objects:
            radius = 0.5 * float(np.hypot(obj.width_m, obj.height_m))
            x, y = obj.position_m
            if fx0 - radius <= x <= fx1 + radius and fy0 - radius <= y <= fy1 + radius:
                hits.append(obj)
        return tuple(hits)

    # -- rendering ----------------------------------------------------------------------

    def render(
        self,
        rgb: np.ndarray,
        footprint_m: tuple[float, float, float, float],
        valid: np.ndarray,
    ) -> RenderedObjects:
        """Paint visible objects into ``rgb`` (in place) and return the ground truth.

        ``rgb`` is the crop already resampled to output resolution; the footprint gives
        its world extent, from which the per-pixel world coordinates follow. Objects are
        only painted onto valid raster (an object cannot stand on nodata scenery).
        """
        h, w = rgb.shape[:2]
        fx0, fy0, fx1, fy1 = footprint_m
        # World coordinates of each output pixel centre.
        xs = fx0 + (np.arange(w) + 0.5) * (fx1 - fx0) / w
        ys = fy0 + (np.arange(h) + 0.5) * (fy1 - fy0) / h
        world_x = xs[None, :]
        world_y = ys[:, None]

        semantic = np.zeros((h, w), dtype=np.uint8)
        instance = np.zeros((h, w), dtype=np.int32)
        visible_targets: list[str] = []
        visible_distractors: list[str] = []

        for obj in self.intersecting(footprint_m):
            inside, stripe = _object_masks(obj, world_x, world_y)
            inside &= valid
            if not inside.any():
                continue

            appearance = obj.appearance
            body = np.array(appearance.get("body_rgb") or (220, 60, 20), dtype=np.float32)
            alpha = float(appearance.get("alpha") or 1.0)
            paint = np.zeros((h, w, 3), dtype=np.float32)
            paint[inside] = body
            if appearance.get("stripe") and appearance.get("stripe_rgb") is not None:
                paint[stripe & inside] = np.array(appearance["stripe_rgb"], dtype=np.float32)

            blended = rgb.astype(np.float32)
            blended[inside] = (1.0 - alpha) * blended[inside] + alpha * paint[inside]
            rgb[inside] = np.clip(blended[inside], 0, 255).astype(np.uint8)

            label = SEMANTIC_TARGET if obj.is_target else SEMANTIC_DISTRACTOR
            semantic[inside] = label
            instance[inside] = self._instance_ids[obj.object_id]
            (visible_targets if obj.is_target else visible_distractors).append(obj.object_id)

        return RenderedObjects(
            semantic=semantic,
            instance=instance,
            visible_target_ids=tuple(visible_targets),
            visible_distractor_ids=tuple(visible_distractors),
        )


def _object_masks(
    obj: ObjectSpec, world_x: np.ndarray, world_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel inside/stripe tests for one object, in its rotated local frame."""
    theta = np.deg2rad(obj.rotation_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    dx = world_x - obj.position_m[0]
    dy = world_y - obj.position_m[1]
    # Inverse-rotate the world offset into the object's axis-aligned local frame.
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    half_w, half_h = obj.width_m / 2.0, obj.height_m / 2.0

    if obj.appearance.get("shape") == "ellipse":
        inside = (local_x / half_w) ** 2 + (local_y / half_h) ** 2 <= 1.0
    else:
        inside = (np.abs(local_x) <= half_w) & (np.abs(local_y) <= half_h)
    # The stripe is a central band across the short axis -- the "rescue marker" pattern.
    stripe = np.abs(local_y) <= (half_h / 3.0)
    return inside, stripe
