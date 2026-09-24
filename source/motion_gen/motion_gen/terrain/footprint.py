# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where a tile's clips actually put the robot's feet, in the tile's own frame.

Every terrain edit in the decoration pass is sized against this. Narrowing a landing is only
safe if the feet still land on it; widening a tread is only safe if the new slab does not
push through geometry the motion already occupies. Both questions are the same question --
what does this tile's motion cover -- and neither can be answered from the manifest, which
describes the ground and not what walks on it.

Three things make the naive answer wrong, and each one has bitten:

* **The clip is baked in the tile's LOCAL-CORNER frame**, so its world channels are already
  tile-local and need no transform. Applying one would silently shift every measurement.
* **A foot is not a point.** ``ankle_roll_link:pos_y`` under-reports the real lateral
  envelope by 0.10 to 0.13 m, because the foot is a 0.203 x 0.065 m box that YAWS: at 45 deg
  its corner reaches 0.083 to 0.092 m off the ankle rather than the 0.033 m half-width.
  So corners are rotated by the recorded quaternion, not assumed axis-aligned.
* **The reference is EGO-ANCHORED laterally at runtime.** The stairs and box anchor masks
  free the lateral DOF and the teacher re-anchors the clip onto the robot's actual root every
  12 steps, so the baked lateral position is a starting point, not where the foot goes.
  Measured drift from the reset placement: p50 0.033, p90 0.110, p99 0.210, max 0.343 m.
  :data:`EGO_ANCHOR_ALLOWANCE_M` carries that, and any envelope used to SIZE terrain must
  include it -- a tread fitted to the baked footprint alone is a tread the robot walks off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Foot collision box in the ankle frame, from ``clip_output``: centre then half-extents (m).
#: Duplicated rather than imported to keep this module's import surface to numpy; the parity
#: is pinned by ``tests/test_tile_footprint.py``.
FOOT_BOX_CENTER = (0.03592, 0.0, -0.02517)
FOOT_BOX_HALF = (0.101555, 0.032735, 0.009255)

#: Lateral margin added per side when SIZING terrain, in metres.
#:
#: The p99 of the measured ego-anchor drift (0.210 m), rounded up. Not the max (0.343 m):
#: sizing every tile for the worst step ever recorded would refuse most of the narrowing this
#: exists to enable, and the tail is already covered by the terrain class's own ground block.
#: Not the p90 either -- 0.110 m would put roughly one step in ten off the edge.
EGO_ANCHOR_ALLOWANCE_M = 0.30


@dataclass(frozen=True)
class TileFootprint:
    """The lateral and downrange extent one tile's clips cover.

    Attributes:
        terrain_id: The tile.
        n_clips: Clips assigned to it.
        y_min: Smallest foot-corner y, in the tile frame (m).
        y_max: Largest foot-corner y (m).
        x_min: Smallest foot-corner x (m).
        x_max: Largest foot-corner x (m).
        y_min_sized: ``y_min`` less :data:`EGO_ANCHOR_ALLOWANCE_M`.
        y_max_sized: ``y_max`` plus :data:`EGO_ANCHOR_ALLOWANCE_M`.
    """

    terrain_id: int
    n_clips: int
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    y_min_sized: float
    y_max_sized: float
    # The same envelope over the TRAVERSAL frames only (``anchor_mask_t == 2``), i.e. the feet that
    # are actually on the terrain. A flight has to carry those and nothing else; the whole-clip
    # envelope also contains the approach and exit, which happen on the flat PLATFORMS.
    #
    # Sizing a flight from the whole clip is why the shipped stairs are as wide as they are: measured
    # on the v6b build, the whole-clip union over a tile's 50 clips is 2.51 m median while the
    # traversal-only union is 0.35 m (max 0.56). The difference is not the robot wandering -- one
    # clip spans 0.61 m -- it is the 50 clips being laterally offset from each other by ~1 m on the
    # platforms, which a staircase has no reason to span.
    #
    # ``None`` where a caller could not compute it (a library with no anchor mask); consumers then
    # fall back to the whole-clip envelope, which is the safe direction.
    tread_y_min_sized: float | None = None
    tread_y_max_sized: float | None = None

    def tread_envelope(self) -> tuple:
        """``(lo, hi)`` a FLIGHT must span: the traversal feet if known, else the whole clip."""
        if self.tread_y_min_sized is None or self.tread_y_max_sized is None:
            return (self.y_min_sized, self.y_max_sized)
        return (float(self.tread_y_min_sized), float(self.tread_y_max_sized))

    @property
    def width(self) -> float:
        """Lateral extent actually covered, before the allowance (m)."""
        return self.y_max - self.y_min

    @property
    def width_sized(self) -> float:
        """Lateral extent a surface must span to be safe (m)."""
        return self.y_max_sized - self.y_min_sized


def foot_corners(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World positions of a foot box's eight corners.

    Args:
        pos: ``(..., 3)`` ankle positions.
        quat_wxyz: ``(..., 4)`` ankle orientations, w first.

    Returns:
        ``(..., 8, 3)`` corner positions in the same frame as ``pos``.

    Note:
        All eight, not the four at sole height: a tread has to clear the whole box, and the
        top corners are what a widened slab would intersect.
    """
    cx, cy, cz = FOOT_BOX_CENTER
    hx, hy, hz = FOOT_BOX_HALF
    local = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                      for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])

    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return np.einsum("...ij,kj->...ki", rot, local) + np.asarray(pos)[..., None, :]


def tile_footprint(ankle_pos: np.ndarray, ankle_quat: np.ndarray, terrain_id: int,
                   n_clips: int, allowance: float = EGO_ANCHOR_ALLOWANCE_M,
                   tread_pos: np.ndarray | None = None,
                   tread_quat: np.ndarray | None = None) -> TileFootprint:
    """Reduce one tile's foot samples to an envelope.

    Args:
        ankle_pos: ``(N, 3)`` ankle positions, tile frame, both feet pooled.
        ankle_quat: ``(N, 4)`` ankle orientations, w first.
        terrain_id: The tile these came from.
        n_clips: How many clips contributed.
        allowance: Lateral margin per side (m).

    Returns:
        The envelope.

    Raises:
        ValueError: If no samples were given -- an empty envelope would read as a tile that
            needs no surface at all, which is the most dangerous possible answer.
    """
    if len(ankle_pos) == 0:
        raise ValueError(
            f"terrain {terrain_id} has no foot samples. An empty envelope would size the "
            "tile to nothing; check the clip selection rather than trusting this."
        )
    corners = foot_corners(ankle_pos, ankle_quat).reshape(-1, 3)
    y_min, y_max = float(corners[:, 1].min()), float(corners[:, 1].max())
    return TileFootprint(
        terrain_id=terrain_id, n_clips=n_clips,
        y_min=y_min, y_max=y_max,
        x_min=float(corners[:, 0].min()), x_max=float(corners[:, 0].max()),
        y_min_sized=y_min - allowance, y_max_sized=y_max + allowance,
        tread_y_min_sized=(None if tread_pos is None or len(tread_pos) == 0
                           else float(foot_corners(tread_pos, tread_quat).reshape(-1, 3)[:, 1].min())
                           - allowance),
        tread_y_max_sized=(None if tread_pos is None or len(tread_pos) == 0
                           else float(foot_corners(tread_pos, tread_quat).reshape(-1, 3)[:, 1].max())
                           + allowance),
    )
